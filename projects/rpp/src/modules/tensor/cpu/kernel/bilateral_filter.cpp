/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "host_tensor_executors.hpp"

/*
bilateral_filter -- HOST kernel

Spec (the gtest golden, utilities/test/src/reference/bilateral_filter_ref.hpp, is the authority):

  For every image n, every channel c, every output pixel (j, i) of the ROI:

    r  = kernelSize / 2
    v0 = src(y0 + j, x0 + i)                                          centre pixel
    v  = src(y0 + clampY(j + dy), x0 + clampX(i + dx))                for dy, dx in [-r, r]
    w  = exp(-(dy^2 + dx^2) / (2 * stdDevSpaceTensor[n]^2))
       * exp(-(v - v0)^2    / (2 * stdDevRangeTensor[n]^2))
    dst(j, i) = sum(w * v) / sum(w)

  - Square KxK window (all K^2 taps), kernelSize in {3, 5, 7, 9}.
  - Each channel is filtered on its own: v, v0 are values of the SAME channel.
  - REPLICATE border clamped to the ROI, not the image: clampY to [0, roiHeight - 1],
    clampX to [0, roiWidth - 1]. Pixels outside the ROI are never read.
  - Source is read at the ROI offset (x0, y0); output is written from the destination origin.
  - Standard deviations are per image (index by batchCount, not [0]).
  - stdDevRangeTensor is in the stored units, so (v - v0) needs no rescaling for any dtype; the
    I8 -128 shift cancels in the difference.
  - Store: U8/I8 round to nearest then clamp ([0,255] / [-128,127]); F16/F32 clamp to [0,1].
  - Layouts: srcDescPtr / dstDescPtr may differ (NHWC <-> NCHW); address src through src strides
    and dst through dst strides.

  Reference structure for the batch loop: gaussian_filter_host_tensor in gaussian_filter.cpp
  (compute_roi_validation_host, nStride per image, omp parallel for over the batch).

  Test: rpp_tests --gtest_filter='Image_Filter/BilateralFilter*HOST_*'
*/

namespace rpp_bilateral_filter {

inline void create_bilateral_spatial_kernel_host(Rpp32f* spatialKernel, Rpp32f stdDevSpace,
                                                 int kernelSize) {
    int kernelHalfSize = kernelSize / 2;
    // double: in float, 2 * sigma^2 underflows for sigma below ~4e-20 and the factor becomes inf,
    // so the centre tap's 0 * inf gives NaN instead of a weight of 1
    Rpp64f mulFactor = 1.0 / (2.0 * (Rpp64f)stdDevSpace * stdDevSpace);

    for (int i = -kernelHalfSize; i <= kernelHalfSize; i++) {
        for (int j = -kernelHalfSize; j <= kernelHalfSize; j++) {
            spatialKernel[(i + kernelHalfSize) * kernelSize + (j + kernelHalfSize)] =
                std::exp(-(i * i + j * j) * mulFactor);
        }
    }
}

inline void create_bilateral_range_host(Rpp32f* range, Rpp32f stdDevColor) {
    Rpp64f mulFactor = 1.0 / (2.0 * (Rpp64f)stdDevColor * stdDevColor);  // see spatial kernel

    for (int i = 0; i < 256; i++) {
        range[i] = std::exp(-(i * i) * mulFactor);
    }
}

template <typename T>
inline void bilateral_store_result(Rpp32f sum, Rpp32f weightSum, T* dstPtrTemp) {
    Rpp32f result = sum / weightSum;
    if (typeid(T) == typeid(Rpp8s)) {
        result += 128;  // saturate_pixel subtracts 128 before clamping for Rpp8s
    }
    if (typeid(T) == typeid(Rpp8u) || typeid(T) == typeid(Rpp8s)) {
        result = nearbyintf(result);
    }
    saturate_pixel(result, dstPtrTemp);
}

// Filters one output pixel. srcPtrRows[k] points at ROI column 0 of the source row under window row
// k, already clamped to the ROI vertically. colOffsets[j] is the element offset of window column j,
// already clamped to the ROI horizontally. Clamping both axes to both ROI edges replicates the
// border even when the window overhangs opposite edges.
template <typename T>
inline void bilateral_filter_generic_tensor(T** srcPtrRows, T* dstPtrTemp, Rpp32s* colOffsets,
                                            Rpp32u kernelSize, Rpp32s padLength,
                                            Rpp32f* spatialKernel, Rpp32f* range,
                                            Rpp32f rangeMulFactor) {
    Rpp32f center = srcPtrRows[padLength][colOffsets[padLength]];
    Rpp32f sum = 0.0f, weightSum = 0.0f;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            Rpp32f p = srcPtrRows[i][colOffsets[j]];
            Rpp32f diff = p - center;
            Rpp32f rangeWeight;
            if (typeid(T) == typeid(Rpp8u) || typeid(T) == typeid(Rpp8s)) {
                rangeWeight = range[(Rpp32s)std::abs(diff)];
            } else {
                rangeWeight = std::exp(-(diff * diff) * rangeMulFactor);
            }
            Rpp32f w = spatialKernel[i * kernelSize + j] * rangeWeight;
            sum += w * p;
            weightSum += w;
        }
    }

    bilateral_store_result(sum, weightSum, dstPtrTemp);
}

}  // namespace rpp_bilateral_filter

using namespace rpp_bilateral_filter;

// -------------------- Per-image generic (scalar) implementation --------------------

// Follows median_filter: for every output row the kernelSize source row pointers are rebuilt and
// clamped to the ROI, and the clamped column offsets are tabulated once per image, so there is no
// separate top/bottom or left/right border pass and an ROI smaller than the window is handled like
// any other. Every layout pair and channel count is addressed through the descriptor strides.
template <typename T>
static inline RppStatus bilateral_filter_generic_host_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                           T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                           Rpp32f* spatialKernel, Rpp32f* rangeLut,
                                                           Rpp32f rangeMulFactor, Rpp32u kernelSize,
                                                           RpptROI roi) {
    Rpp32s padLength = kernelSize / 2;
    Rpp32s roiWidth = roi.xywhROI.roiWidth;
    Rpp32s roiHeight = roi.xywhROI.roiHeight;
    T* srcPtrRoi = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                   (roi.xywhROI.xy.x * srcDescPtr->strides.wStride);

    // colOffsets[j + k] is the source offset of window column k for output column j
    std::vector<Rpp32s> colOffsets(roiWidth + 2 * padLength);
    for (int j = 0; j < colOffsets.size(); j++)
        colOffsets[j] =
            std::min(std::max(j - padLength, 0), roiWidth - 1) * srcDescPtr->strides.wStride;

    T* srcPtrRows[kernelSize];
    for (int i = 0; i < roiHeight; i++) {
        for (int c = 0; c < srcDescPtr->c; c++) {
            T* srcPtrChannel = srcPtrRoi + c * srcDescPtr->strides.cStride;
            for (int k = 0; k < kernelSize; k++) {
                Rpp32s row = std::min(std::max(i - padLength + k, 0), roiHeight - 1);
                srcPtrRows[k] = srcPtrChannel + row * srcDescPtr->strides.hStride;
            }
            T* dstPtrTemp =
                dstPtrImage + c * dstDescPtr->strides.cStride + i * dstDescPtr->strides.hStride;
            for (int j = 0; j < roiWidth; j++) {
                bilateral_filter_generic_tensor(srcPtrRows, dstPtrTemp, colOffsets.data() + j,
                                                kernelSize, padLength, spatialKernel, rangeLut,
                                                rangeMulFactor);
                dstPtrTemp += dstDescPtr->strides.wStride;
            }
        }
    }
    return RPP_SUCCESS;
}

template <typename T>
RppStatus bilateral_filter_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                       Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* spaceTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost +
                              batchCount * kernelSize * kernelSize;
        create_bilateral_spatial_kernel_host(spaceTensor, stdDevSpaceTensor[batchCount],
                                             kernelSize);

        Rpp32f range[256];
        create_bilateral_range_host(range, stdDevRangeTensor[batchCount]);
        // Computed in double, then saturated to the largest float so the per-tap float exponent
        // sees 0 * FLT_MAX = 0 (weight 1) at diff == 0 rather than 0 * inf = NaN for a tiny sigma.
        Rpp64f rangeStdDev = stdDevRangeTensor[batchCount];
        Rpp32f rangeMulFactor = std::min(1.0 / (2.0 * rangeStdDev * rangeStdDev),
                                         (Rpp64f)std::numeric_limits<Rpp32f>::max());

        bilateral_filter_generic_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                           spaceTensor, range, rangeMulFactor, kernelSize, roi);
    }
    return RPP_SUCCESS;
}

template RppStatus bilateral_filter_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                       Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                       RpptRoiType, RppLayoutParams, rpp::Handle&);
template RppStatus bilateral_filter_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                        Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                        RpptRoiType, RppLayoutParams, rpp::Handle&);
template RppStatus bilateral_filter_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*, RpptDescPtr,
                                                        Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                        RpptRoiType, RppLayoutParams, rpp::Handle&);
template RppStatus bilateral_filter_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                       Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                       RpptRoiType, RppLayoutParams, rpp::Handle&);

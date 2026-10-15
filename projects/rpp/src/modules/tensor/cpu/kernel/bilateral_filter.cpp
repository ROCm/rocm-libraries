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
#include "rpp_cpu_filter.hpp"

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
    and dst through dst strides. layoutParams carries the channel/pixel increments.

  Reference structure for the batch loop: gaussian_filter_host_tensor in gaussian_filter.cpp
  (compute_roi_validation_host, nStride per image, omp parallel for over the batch).

  Test: rpp_tests --gtest_filter='Image_Filter/BilateralFilter*HOST_*'
*/

namespace rpp_bilateral_filter {

inline void create_bilateral_spatial_kernel_host(Rpp32f* spatialKernel, Rpp32f stdDevSpace,
                                                 int kernelSize) {
    int kernelHalfSize = kernelSize / 2;
    Rpp32f mulFactor = 1.0f / (2.0f * stdDevSpace * stdDevSpace);

    for (int i = -kernelHalfSize; i <= kernelHalfSize; i++) {
        for (int j = -kernelHalfSize; j <= kernelHalfSize; j++) {
            spatialKernel[(i + kernelHalfSize) * kernelSize + (j + kernelHalfSize)] =
                std::exp(-(i * i + j * j) * mulFactor);
        }
    }
}

inline void create_bilateral_range_host(Rpp32f* range, Rpp32f stdDevColor) {
    Rpp32f mulFactor = 1.0f / (2.0f * stdDevColor * stdDevColor);

    for (int i = 0; i < 256; i++) {
        range[i] = std::exp(-(i * i) * mulFactor);
    }
}

template <typename T>
inline void bilateral_filter_generic_tensor(
    T** srcPtrTemp, T* dstPtrTemp, Rpp32s columnIndex, Rpp32u kernelSize, Rpp32u padLength,
    Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f* spatialKernel, Rpp32f* range,
    Rpp32f rangeMulFactor, Rpp32u channels = 1,
    RpptImageBorderEdge padVertical = RpptImageBorderEdge::BOTTOM_EDGE,
    RpptImageBorderEdge padHorizontal = RpptImageBorderEdge::RIGHT_EDGE) {
    Rpp32s columnKernelLoopLimit = kernelSize;
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);

    auto clampRow = [&](Rpp32s r) {
        return (padVertical == RpptImageBorderEdge::TOP_EDGE)
                   ? std::max(0, (Rpp32s)(r + rowKernelLoopLimit - kernelSize))
                   : std::min(rowKernelLoopLimit - 1, r);
    };
    auto clampCol = [&](Rpp32s c) {
        return (padHorizontal == RpptImageBorderEdge::LEFT_EDGE)
                   ? std::max(0, (Rpp32s)(c + columnKernelLoopLimit - kernelSize))
                   : std::min(columnKernelLoopLimit - 1, c);
    };

    Rpp32f center = srcPtrTemp[clampRow(padLength)][clampCol(padLength) * channels];
    Rpp32f sum = 0.0f, weightSum = 0.0f;
    for (int i = 0; i < kernelSize; i++) {
        Rpp32s row = clampRow(i);
        for (int j = 0; j < kernelSize; j++) {
            Rpp32f p = srcPtrTemp[row][clampCol(j) * channels];
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

    Rpp32f result = sum / weightSum;
    if (typeid(T) == typeid(Rpp8s)) {
        result += 128;  // saturate_pixel subtracts 128 before clamping for Rpp8s
    }
    if (typeid(T) == typeid(Rpp8u) || typeid(T) == typeid(Rpp8s)) {
        result = nearbyintf(result);
    }
    saturate_pixel(result, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them
// separately
template <typename T>
inline void bilateral_process_left_border_columns_pln_pln(
    T** srcPtrTemp, T* dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth,
    Rpp32s rowKernelLoopLimit, Rpp32f* spatialKernel, Rpp32f* rangeLut, Rpp32f rangeMulFactor,
    RpptImageBorderEdge padVertical) {
    for (int k = 0; k < padLength; k++) {
        bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength,
                                        unpaddedWidth, rowKernelLoopLimit, spatialKernel, rangeLut,
                                        rangeMulFactor, 1, padVertical,
                                        RpptImageBorderEdge::LEFT_EDGE);
        dstPtrTemp++;
    }
}

template <typename T>
inline void bilateral_process_left_border_columns_pkd_pkd(
    T** srcPtrTemp, T** srcPtrRow, T* dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
    Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f* spatialKernel, Rpp32f* rangeLut,
    Rpp32f rangeMulFactor, RpptImageBorderEdge padVertical) {
    for (int c = 0; c < 3; c++) {
        T* dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++) {
            bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength,
                                            unpaddedWidth, rowKernelLoopLimit, spatialKernel,
                                            rangeLut, rangeMulFactor, 3, padVertical,
                                            RpptImageBorderEdge::LEFT_EDGE);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }
    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

template <typename T>
inline void bilateral_process_left_border_columns_pkd_pln(
    T** srcPtrTemp, T** srcPtrRow, T** dstPtrTempChannels, Rpp32u kernelSize, Rpp32u padLength,
    Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f* spatialKernel, Rpp32f* rangeLut,
    Rpp32f rangeMulFactor, RpptImageBorderEdge padVertical) {
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < padLength; k++) {
            bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize,
                                            padLength, unpaddedWidth, rowKernelLoopLimit,
                                            spatialKernel, rangeLut, rangeMulFactor, 3, padVertical,
                                            RpptImageBorderEdge::LEFT_EDGE);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

}  // namespace rpp_bilateral_filter

using namespace rpp_bilateral_filter;

// -------------------- Per-image generic (scalar) implementation --------------------

template <typename T>
static inline RppStatus bilateral_filter_generic_host_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                           T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                           Rpp32f* spatialKernel, Rpp32f* rangeLut,
                                                           Rpp32f rangeMulFactor, Rpp32u kernelSize,
                                                           RpptROI roi,
                                                           RppLayoutParams layoutParams) {
    Rpp32u padLength = kernelSize / 2;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
    Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    T *srcPtrRow[kernelSize], *dstPtrRow;
    for (int k = 0; k < kernelSize; k++)
        srcPtrRow[k] = srcPtrChannel + k * srcDescPtr->strides.hStride;
    dstPtrRow = dstPtrChannel;

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        for (int c = 0; c < srcDescPtr->c; c++) {
            srcPtrRow[0] = srcPtrChannel;
            for (int k = 1; k < kernelSize; k++)
                srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength);
                T* srcPtrTemp[kernelSize];
                for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                bilateral_process_left_border_columns_pln_pln(
                    srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                    rowKernelLoopLimit, spatialKernel, rangeLut, rangeMulFactor, padVertical);
                dstPtrTemp += padLength;
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, spatialKernel, rangeLut,
                                                    rangeMulFactor, 1, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
            T* dstPtrTemp = dstPtrRow;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            bilateral_process_left_border_columns_pkd_pkd(
                srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                rowKernelLoopLimit, spatialKernel, rangeLut, rangeMulFactor, padVertical);
            dstPtrTemp += padLength * 3;
            vectorLoopCount += padLength * 3;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3,
                                                kernelSize, padLength, unpaddedWidth,
                                                rowKernelLoopLimit, spatialKernel, rangeLut,
                                                rangeMulFactor, 3, padVertical);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTemp++;
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[3][kernelSize];
            for (int c = 0; c < 3; c++) {
                Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
            }
            T* dstPtrTemp = dstPtrRow;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            for (int k = 0; k < padLength; k++) {
                for (int c = 0; c < 3; c++) {
                    bilateral_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize,
                                                    padLength, unpaddedWidth, rowKernelLoopLimit,
                                                    spatialKernel, rangeLut, rangeMulFactor, 1,
                                                    padVertical, RpptImageBorderEdge::LEFT_EDGE);
                    dstPtrTemp++;
                }
            }
            vectorLoopCount += padLength;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                for (int c = 0; c < srcDescPtr->c; c++) {
                    bilateral_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, spatialKernel, rangeLut,
                                                    rangeMulFactor, 1, padVertical);
                    increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                    dstPtrTemp++;
                }
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        T* dstPtrChannels[3];
        for (int c = 0; c < 3; c++)
            dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
            T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            bilateral_process_left_border_columns_pkd_pln(
                srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth,
                rowKernelLoopLimit, spatialKernel, rangeLut, rangeMulFactor, padVertical);
            vectorLoopCount += padLength * 3;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                int channel = vectorLoopCount % 3;
                bilateral_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                                vectorLoopCount / 3, kernelSize, padLength,
                                                unpaddedWidth, rowKernelLoopLimit, spatialKernel,
                                                rangeLut, rangeMulFactor, 3, padVertical);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTempChannels[channel]++;
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
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
        Rpp32f rangeMulFactor =
            1.0f / (2.0f * stdDevRangeTensor[batchCount] * stdDevRangeTensor[batchCount]);

        bilateral_filter_generic_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                           spaceTensor, range, rangeMulFactor, kernelSize, roi,
                                           layoutParams);
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

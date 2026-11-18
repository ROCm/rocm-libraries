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
prewitt_filter -- HOST kernel

Spec (the gtest golden, utilities/test/src/reference/prewitt_filter_ref.hpp, is the authority):

  For every image n, every channel c, every output pixel (j, i) of the ROI:

    w(dy, dx) = src(y0 + clampY(j + dy), x0 + clampX(i + dx))       for dy, dx in [-1, 1]
    gx = sum Gx * w,  Gx = [-1 0 1; -1 0 1; -1 0 1]                 (row-major, dy outer)
    gy = sum Gy * w,  Gy = [-1 -1 -1; 0 0 0; 1 1 1]
    dst(j, i) = prewittType == 0 ? gx : prewittType == 1 ? gy : sqrt(gx^2 + gy^2)

  - 3x3 only; there is no kernelSize argument.
  - Each channel is filtered on its own: the window holds values of the SAME channel.
  - REPLICATE border clamped to the ROI, not the image: clampY to [0, roiHeight - 1],
    clampX to [0, roiWidth - 1]. Pixels outside the ROI are never read.
  - Source is read at the ROI offset (x0, y0); output is written from the destination origin.
  - XY magnitude uses the UNCLAMPED gx and gy. Clamp only the final result. (sobel_filter.cpp
    clamps gx/gy before squaring; that is a known defect, don't copy it.)
  - I8: the gradient is in intensity space (stored + 128). Gx and Gy sum to zero, so gx/gy computed
    on stored values are already correct; subtract 128 from the result before the store. A flat
    region therefore stores -128, as U8 stores 0.
  - Store: U8/I8 round to nearest then clamp ([0,255] / [-128,127]); F16/F32 clamp to [0,1].
    Negative gradients therefore saturate to the range minimum.
  - Layouts: srcDescPtr / dstDescPtr may differ (NHWC <-> NCHW); address src through src strides
    and dst through dst strides. layoutParams carries the source channel/pixel increments.

  Implementation: scalar, one omp thread per image of the batch. No AVX2 path yet; the 3x3
  load/permute helpers sobel_filter.cpp vectorizes with are in rpp_cpu_filter.hpp.

  Test: rpp_tests --gtest_filter='Image_Filter/PrewittFilter*HOST_*'
*/

template <typename T>
RppStatus prewitt_filter_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u prewittType,
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

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // ROI bounds in image coordinates
        Rpp32s xStart = roi.xywhROI.xy.x;
        Rpp32s yStart = roi.xywhROI.xy.y;
        Rpp32s xEnd = xStart + roi.xywhROI.roiWidth;
        Rpp32s yEnd = yStart + roi.xywhROI.roiHeight;

        // Each channel is filtered on its own; strides handle both NCHW (planar) and NHWC (packed)
        for (Rpp32u c = 0; c < srcDescPtr->c; c++) {
            T* srcPtrChannel = srcPtrImage + c * srcDescPtr->strides.cStride;
            T* dstPtrChannel = dstPtrImage + c * dstDescPtr->strides.cStride;

            // REPLICATE border: neighbors outside the ROI take the nearest ROI edge pixel, so
            // padding is never read. Gx and Gy sum to zero, so an i8 pixel needs no +128 shift here
            auto srcPixel = [&](Rpp32s y, Rpp32s x) -> Rpp32f {
                y = std::clamp(y, yStart, yEnd - 1);
                x = std::clamp(x, xStart, xEnd - 1);
                return static_cast<Rpp32f>(srcPtrChannel[y * srcDescPtr->strides.hStride +
                                                         x * srcDescPtr->strides.wStride]);
            };

            for (Rpp32s y = yStart; y < yEnd; y++) {
                T* dstPtrRow = dstPtrChannel + (y - yStart) * dstDescPtr->strides.hStride;
                for (Rpp32s x = xStart; x < xEnd; x++) {
                    // The 8 neighbors of (y, x); the center has weight 0 in both kernels
                    Rpp32f topLeft = srcPixel(y - 1, x - 1);
                    Rpp32f top = srcPixel(y - 1, x);
                    Rpp32f topRight = srcPixel(y - 1, x + 1);
                    Rpp32f left = srcPixel(y, x - 1);
                    Rpp32f right = srcPixel(y, x + 1);
                    Rpp32f bottomLeft = srcPixel(y + 1, x - 1);
                    Rpp32f bottom = srcPixel(y + 1, x);
                    Rpp32f bottomRight = srcPixel(y + 1, x + 1);

                    // Gx: right column minus left column. Gy: bottom row minus top row
                    Rpp32f gx = (topRight + right + bottomRight) - (topLeft + left + bottomLeft);
                    Rpp32f gy = (bottomLeft + bottom + bottomRight) - (topLeft + top + topRight);

                    // prewittType 0 = X, 1 = Y, 2 = XY. The magnitude uses the unclamped gx and
                    // gy; only the final value is saturated to the destination range
                    Rpp32f result = (prewittType == 0)   ? gx
                                    : (prewittType == 1) ? gy
                                                         : std::sqrt(gx * gx + gy * gy);

                    // saturate_pixel rounds and clamps u8/i8 (subtracting 128 for i8) and clamps
                    // f16/f32 to [0, 1]
                    saturate_pixel(result, dstPtrRow + (x - xStart) * dstDescPtr->strides.wStride);
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus prewitt_filter_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                     Rpp32u, RpptROIPtr, RpptRoiType,
                                                     RppLayoutParams, rpp::Handle&);
template RppStatus prewitt_filter_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                      Rpp32u, RpptROIPtr, RpptRoiType,
                                                      RppLayoutParams, rpp::Handle&);
template RppStatus prewitt_filter_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*, RpptDescPtr,
                                                      Rpp32u, RpptROIPtr, RpptRoiType,
                                                      RppLayoutParams, rpp::Handle&);
template RppStatus prewitt_filter_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                     Rpp32u, RpptROIPtr, RpptRoiType,
                                                     RppLayoutParams, rpp::Handle&);

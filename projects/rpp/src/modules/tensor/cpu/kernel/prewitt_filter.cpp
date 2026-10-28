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

  Reference structure for the batch loop: gaussian_filter_host_tensor in gaussian_filter.cpp
  (compute_roi_validation_host, nStride per image, omp parallel for over the batch). The 3x3
  load/permute helpers sobel_filter.cpp uses are in rpp_cpu_filter.hpp.

  Test: rpp_tests --gtest_filter='Image_Filter/PrewittFilter*HOST_*'
*/

template <typename T>
RppStatus prewitt_filter_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u prewittType,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    // TODO: implement. Replace this whole body.
    return RPP_ERROR_NOT_IMPLEMENTED;
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

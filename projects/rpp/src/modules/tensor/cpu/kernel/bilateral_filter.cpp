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

template <typename T>
RppStatus bilateral_filter_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                       Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    // TODO: implement. Replace this whole body.
    return RPP_ERROR_NOT_IMPLEMENTED;
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

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

#include "hip_tensor_executors.hpp"

// -------------------- Set 0 - bilateral_filter device helpers --------------------

// TODO: device helpers and __global__ kernels go here

// -------------------- Set 1 - bilateral_filter executors --------------------

// Stub: launches nothing yet, so the HIP tests run and fail correctness instead of failing to link.
template <typename T>
RppStatus hip_exec_bilateral_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                           Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle) {
    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_bilateral_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                 RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                                 Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                 rpp::Handle& handle) {
    return RPP_SUCCESS;
}

template RppStatus hip_exec_bilateral_filter_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                           Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                           RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                          Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                          RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                            RpptDescPtr, Rpp32f*, Rpp32f*, Rpp32u,
                                                            RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                           Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                           RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                                 RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                 Rpp32u, RpptROIPtr, RpptRoiType,
                                                                 rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<half>(half*, RpptDescPtr, half*,
                                                                RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                Rpp32u, RpptROIPtr, RpptRoiType,
                                                                rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                                  RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                  Rpp32u, RpptROIPtr, RpptRoiType,
                                                                  rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                                 RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                 Rpp32u, RpptROIPtr, RpptRoiType,
                                                                 rpp::Handle&);

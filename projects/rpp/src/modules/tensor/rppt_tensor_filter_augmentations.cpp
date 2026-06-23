/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "rppt_tensor_filter_augmentations.h"

#include "host_tensor_executors.hpp"
#include "rppdefs.h"
#include "rppt_tensor_data_exchange_operations.h"
#include "rppt_validate.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif  // GPU_SUPPORT

/******************** box_filter ********************/

RppStatus rppt_box_filter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptImageBorderType borderType,
                          RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                          RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (borderType != RpptImageBorderType::REPLICATE) return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                box_filter_char_host_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                box_filter_float_host_single_image(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                box_filter_float_host_single_image(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                box_filter_char_host_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                box_filter_char_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                box_filter_float_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                       srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                       dstDescPtr->offsetInBytes),
                                             dstDescPtr, kernelSize, roiTensorPtrSrc, roiType,
                                             layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                box_filter_float_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                       srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                       dstDescPtr->offsetInBytes),
                                             dstDescPtr, kernelSize, roiTensorPtrSrc, roiType,
                                             layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                box_filter_char_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
            return RPP_ERROR_INVALID_ARGUMENTS;
        if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2)) return RPP_ERROR_LOW_OFFSET;

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_box_filter_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_box_filter_single_image(
                    (half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    kernelSize, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_box_filter_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    kernelSize, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_box_filter_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_box_filter_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_box_filter_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                                   srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                                   dstDescPtr->offsetInBytes),
                                           dstDescPtr, kernelSize, roiTensorPtrSrc, roiType,
                                           handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_box_filter_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                     srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                     dstDescPtr->offsetInBytes),
                                           dstDescPtr, kernelSize, roiTensorPtrSrc, roiType,
                                           handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_box_filter_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** median_filter ********************/

RppStatus rppt_median_filter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                             RpptImageBorderType borderType, RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType, rppHandle_t rppHandle,
                             RppBackend executionBackend) {
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (borderType != RpptImageBorderType::REPLICATE) return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                median_filter_generic_host_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                median_filter_generic_host_single_image(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                median_filter_generic_host_single_image(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                median_filter_generic_host_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                median_filter_generic_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                median_filter_generic_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                median_filter_generic_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                median_filter_generic_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2)) return RPP_ERROR_LOW_OFFSET;

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_median_filter_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_median_filter_single_image(
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                            srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                            dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_median_filter_single_image(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_median_filter_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_median_filter_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_median_filter_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                                      srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                                      dstDescPtr->offsetInBytes),
                                              dstDescPtr, kernelSize, roiTensorPtrSrc, roiType,
                                              handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_median_filter_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_median_filter_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, kernelSize,
                    roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** gaussian_filter ********************/

RppStatus rppt_gaussian_filter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor, Rpp32u kernelSize,
                               RpptImageBorderType borderType, RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType, rppHandle_t rppHandle,
                               RppBackend executionBackend) {
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (borderType != RpptImageBorderType::REPLICATE) return RPP_ERROR_NOT_IMPLEMENTED;

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                gaussian_filter_host_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    *stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                gaussian_filter_host_single_image(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, *stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                    handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                gaussian_filter_host_single_image(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, *stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                    handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                gaussian_filter_host_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    *stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                gaussian_filter_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                gaussian_filter_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                      srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                      dstDescPtr->offsetInBytes),
                                            dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc,
                                            roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                gaussian_filter_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                      srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                      dstDescPtr->offsetInBytes),
                                            dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc,
                                            roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                gaussian_filter_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2)) return RPP_ERROR_LOW_OFFSET;

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_gaussian_filter_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_gaussian_filter_single_image(
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                            srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                            dstDescPtr->offsetInBytes),
                    dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_gaussian_filter_single_image(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_gaussian_filter_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_gaussian_filter_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_gaussian_filter_tensor(
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                            srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                            dstDescPtr->offsetInBytes),
                    dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_gaussian_filter_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_gaussian_filter_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** sobel_filter ********************/

#define SOBEL_TYPE_X_GRADIENT 0
#define SOBEL_TYPE_Y_GRADIENT 1
#define SOBEL_TYPE_XY_GRADIENT 2

RppStatus rppt_sobel_filter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr, Rpp32u sobelType, Rpp32u kernelSize,
                            RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                            RppBackend executionBackend) {
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if ((sobelType != SOBEL_TYPE_X_GRADIENT) && (sobelType != SOBEL_TYPE_Y_GRADIENT) &&
        (sobelType != SOBEL_TYPE_XY_GRADIENT))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NCHW) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (dstDescPtr->c == 3) return RPP_ERROR_INVALID_DST_CHANNELS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        // convert image to grey scale if input is RGB image
        RppPtr_t tempPtr = srcPtr;
        RpptDescPtr inputDesc = srcDescPtr;
        if (srcDescPtr->c == 3) {
            RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
            tempPtr = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
            RppStatus errorStatus =
                rppt_color_to_greyscale(srcPtr, srcDescPtr, tempPtr, dstDescPtr, srcSubpixelLayout,
                                        rppHandle, RppBackend::RPP_HOST_BACKEND);
            if (errorStatus != RPP_SUCCESS) return errorStatus;
            // Greyscale wrote to tempPtr using dstDescPtr's layout/offset; sobel must read with the
            // same descriptor.
            inputDesc = dstDescPtr;
        }

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            return sobel_filter_host_tensor(
                static_cast<Rpp8u*>(tempPtr) + inputDesc->offsetInBytes, inputDesc,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, sobelType,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            return sobel_filter_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(tempPtr) + inputDesc->offsetInBytes),
                inputDesc,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, sobelType, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            return sobel_filter_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(tempPtr) + inputDesc->offsetInBytes),
                inputDesc,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, sobelType, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            return sobel_filter_host_tensor(
                static_cast<Rpp8s*>(tempPtr) + inputDesc->offsetInBytes, inputDesc,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, sobelType,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptDescPtr inputDesc = srcDescPtr;
        void* tempPtr = nullptr;
        if (srcDescPtr->c == 3) {
            size_t elementSize = (srcDescPtr->dataType == RpptDataType::F32)   ? 4
                                 : (srcDescPtr->dataType == RpptDataType::F16) ? 2
                                                                               : 1;
            size_t dataSize = dstDescPtr->strides.nStride * dstDescPtr->n * elementSize;
            RPP_HIP_RETURN_IF_ERROR(hipMallocAsync(&tempPtr, dataSize, handle.GetStream()));

            RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
            RppStatus errorStatus =
                rppt_color_to_greyscale(srcPtr, srcDescPtr, tempPtr, dstDescPtr, srcSubpixelLayout,
                                        rppHandle, RppBackend::RPP_HIP_BACKEND);
            if (errorStatus != RPP_SUCCESS) {
                // Ignore hipFree status to preserve the root cause of the error
                (void)hipFreeAsync(tempPtr, handle.GetStream());
                return errorStatus;
            }
            inputDesc = dstDescPtr;
        }
        srcPtr = (tempPtr == nullptr) ? srcPtr : tempPtr;

        RppStatus status = RPP_SUCCESS;
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            status = hip_exec_sobel_filter_tensor(
                static_cast<Rpp8u*>(srcPtr) + inputDesc->offsetInBytes, inputDesc,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, sobelType,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            status = hip_exec_sobel_filter_tensor(
                (half*)(static_cast<Rpp8u*>(srcPtr) + inputDesc->offsetInBytes), inputDesc,
                (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                sobelType, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            status = hip_exec_sobel_filter_tensor(
                (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + inputDesc->offsetInBytes), inputDesc,
                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                sobelType, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            status = hip_exec_sobel_filter_tensor(
                static_cast<Rpp8s*>(srcPtr) + inputDesc->offsetInBytes, inputDesc,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, sobelType,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else
            status = RPP_ERROR_NOT_IMPLEMENTED;

        if (tempPtr != nullptr) {
            hipError_t freeErr = hipFreeAsync(tempPtr, handle.GetStream());
            if (status == RPP_SUCCESS && freeErr != hipSuccess) return RPP_ERROR_HIP_RUNTIME;
        }

        return status;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** emboss ********************/

RppStatus rppt_emboss(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, Rpp32f* strength, Rpp32u kernelSize,
                      RpptImageBorderType borderType, RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (borderType != RpptImageBorderType::REPLICATE) return RPP_ERROR_NOT_IMPLEMENTED;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            emboss_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                               strength, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                               handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            emboss_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, strength, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            emboss_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, strength, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            emboss_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                               strength, kernelSize, roiTensorPtrSrc, roiType, layoutParams,
                               handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2)) return RPP_ERROR_LOW_OFFSET;

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_emboss_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, strength,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_emboss_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, strength, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_emboss_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, strength, kernelSize, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_emboss_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, strength,
                kernelSize, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

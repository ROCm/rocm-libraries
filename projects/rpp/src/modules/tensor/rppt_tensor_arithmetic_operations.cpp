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

#include "rppt_tensor_arithmetic_operations.h"

#include "host_tensor_executors.hpp"
#include "rppdefs.h"
#include "rppt_validate.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif  // GPU_SUPPORT

/******************** fused_multiply_add_scalar ********************/

RppStatus rppt_fused_multiply_add_scalar(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                         RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32f* mulTensor, Rpp32f* addTensor,
                                         RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType,
                                         rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            fused_multiply_add_scalar_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, mulTensor, addTensor, roiGenericPtrSrc, roiType, layoutParams,
                handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        hip_exec_fused_multiply_add_scalar_tensor(
            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                      srcGenericDescPtr->offsetInBytes),
            srcGenericDescPtr,
            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                      dstGenericDescPtr->offsetInBytes),
            dstGenericDescPtr, roiGenericPtrSrc, mulTensor, addTensor, handle);
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** add_scalar ********************/

RppStatus rppt_add_scalar(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr, Rpp32f* addTensor,
                          RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType,
                          rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            add_scalar_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, addTensor, roiGenericPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        hip_exec_add_scalar_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                             srcGenericDescPtr->offsetInBytes),
                                   srcGenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                             dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr, roiGenericPtrSrc, addTensor, handle);
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** subtract_scalar ********************/

RppStatus rppt_subtract_scalar(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                               RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                               Rpp32f* subtractTensor, RpptROI3DPtr roiGenericPtrSrc,
                               RpptRoi3DType roiType, rppHandle_t rppHandle,
                               RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            subtract_scalar_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, subtractTensor, roiGenericPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        hip_exec_subtract_scalar_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                  srcGenericDescPtr->offsetInBytes),
                                        srcGenericDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                  dstGenericDescPtr->offsetInBytes),
                                        dstGenericDescPtr, roiGenericPtrSrc, subtractTensor,
                                        handle);
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** multiply_scalar ********************/

RppStatus rppt_multiply_scalar(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                               RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                               Rpp32f* mulTensor, RpptROI3DPtr roiGenericPtrSrc,
                               RpptRoi3DType roiType, rppHandle_t rppHandle,
                               RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            multiply_scalar_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, mulTensor, roiGenericPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        hip_exec_multiply_scalar_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                  srcGenericDescPtr->offsetInBytes),
                                        srcGenericDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                  dstGenericDescPtr->offsetInBytes),
                                        dstGenericDescPtr, roiGenericPtrSrc, mulTensor, handle);
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** magnitude ********************/

RppStatus rppt_magnitude(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            magnitude_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            magnitude_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            magnitude_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            magnitude_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_magnitude_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_magnitude_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_magnitude_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_magnitude_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr, roiTensorPtrSrc, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** log ********************/

RppStatus rppt_log(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                   RpptGenericDescPtr dstGenericDescPtr, Rpp32u* roiTensor, rppHandle_t rppHandle,
                   RppBackend executionBackend) {
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
        (dstGenericDescPtr->dataType == RpptDataType::U8))
        return RPP_ERROR_INVALID_DST_DATATYPE;
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))
        return RPP_ERROR_INVALID_DST_DATATYPE;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            log_generic_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                              dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            log_generic_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                              srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                              dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            log_generic_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                              srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                              dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            log_generic_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8s*>(dstPtr) +
                                                              dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_log_generic_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_log_generic_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                                srcGenericDescPtr->offsetInBytes),
                                        srcGenericDescPtr,
                                        reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                                dstGenericDescPtr->offsetInBytes),
                                        dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_log_generic_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                  srcGenericDescPtr->offsetInBytes),
                                        srcGenericDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                  dstGenericDescPtr->offsetInBytes),
                                        dstGenericDescPtr, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_log_generic_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8s*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** log1p ********************/

RppStatus rppt_log1p(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                     RpptGenericDescPtr dstGenericDescPtr, Rpp32u* roiTensor, rppHandle_t rppHandle,
                     RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != RpptDataType::I16) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcGenericDescPtr->dataType == RpptDataType::I16) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            log1p_i16_f32_host_tensor(
                static_cast<Rpp16s*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcGenericDescPtr->dataType == RpptDataType::I16) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_log1p_i16_f32_tensor(
                static_cast<Rpp16s*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp16s*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_add_tensor ********************/

RppStatus rppt_tensor_add_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                 Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                 RppBackend executionBackend) {
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_op_dispatch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else {
            return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if (executionBackend == RppBackend::RPP_HIP_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_ADD, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_subtract_tensor ********************/

RppStatus rppt_tensor_subtract_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                      RpptGenericDescPtr srcPtr1GenericDescPtr,
                                      RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                      Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                      RppBackend executionBackend) {
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_op_dispatch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if (executionBackend == RppBackend::RPP_HIP_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_SUBTRACT, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_multiply_tensor ********************/

RppStatus rppt_tensor_multiply_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                      RpptGenericDescPtr srcPtr1GenericDescPtr,
                                      RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                      Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                      RppBackend executionBackend) {
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_op_dispatch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if (executionBackend == RppBackend::RPP_HIP_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I16)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_MULTIPLY, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_divide_tensor ********************/

RppStatus rppt_tensor_divide_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                    RpptGenericDescPtr srcPtr1GenericDescPtr,
                                    RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                    Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                    RppBackend executionBackend) {
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_DATATYPE;

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_op_dispatch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_bitwise_op_dispatch_int_host_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if (executionBackend == RppBackend::RPP_HIP_BACKEND) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            tensor_binary_arithmetic_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_DIVIDE, broadcastMode, roiTensorSrc1,
                roiTensorSrc2, rpp::deref(rppHandle));
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

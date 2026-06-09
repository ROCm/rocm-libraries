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

#include "rppdefs.h"
#include "rppt_validate.hpp"
#include "rppt_tensor_color_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif // GPU_SUPPORT

/******************** brightness ********************/

RppStatus rppt_brightness(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *alphaTensor,
                          Rpp32f *betaTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1)
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                brightness_u8_u8_host_single_image(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                brightness_f16_f16_host_single_image((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    alphaTensor,
                                                    betaTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    layoutParams,
                                                    rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                brightness_f32_f32_host_single_image((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    alphaTensor,
                                                    betaTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    layoutParams,
                                                    rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                brightness_i8_i8_host_single_image(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                brightness_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            alphaTensor,
                                            betaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                brightness_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            alphaTensor,
                                            betaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                brightness_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            alphaTensor,
                                            betaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                brightness_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            alphaTensor,
                                            betaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1)
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                hip_exec_brightness_single_image(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                hip_exec_brightness_single_image((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                hip_exec_brightness_single_image((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                hip_exec_brightness_single_image(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                alphaTensor,
                                                betaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                hip_exec_brightness_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        alphaTensor,
                                        betaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                hip_exec_brightness_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        alphaTensor,
                                        betaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                hip_exec_brightness_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        alphaTensor,
                                        betaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                hip_exec_brightness_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        alphaTensor,
                                        betaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** gamma_correction ********************/

RppStatus rppt_gamma_correction(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *gammaTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle,
                                RppBackend executionBackend)
{
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            gamma_correction_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               gammaTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            gamma_correction_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                 srcDescPtr,
                                                 reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 gammaTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            gamma_correction_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                 srcDescPtr,
                                                 reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 gammaTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            gamma_correction_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               gammaTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_gamma_correction_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_gamma_correction_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_gamma_correction_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_gamma_correction_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** blend ********************/

RppStatus rppt_blend(RppPtr_t srcPtr1,
                     RppPtr_t srcPtr2,
                     RpptDescPtr srcDescPtr,
                     RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr,
                     Rpp32f *alphaTensor,
                     RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType,
                     rppHandle_t rppHandle,
                     RppBackend executionBackend)
{

    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1)
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                blend_u8_u8_host_single_image(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                              static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              alphaTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams,
                                              handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                blend_f16_f16_host_single_image(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                alphaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                blend_f32_f32_host_single_image(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                alphaTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                blend_i8_i8_host_single_image(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                              static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              alphaTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams,
                                              handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                blend_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        alphaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                blend_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        alphaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                blend_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        alphaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                blend_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        alphaTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1)
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                hip_exec_blend_single_image(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                            static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            alphaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                hip_exec_blend_single_image(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                            reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            alphaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                hip_exec_blend_single_image(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            alphaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                hip_exec_blend_single_image(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                            static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            alphaTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
            {
                hip_exec_blend_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      alphaTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
            {
                hip_exec_blend_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                      reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      alphaTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
            {
                hip_exec_blend_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      alphaTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
            }
            else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
            {
                hip_exec_blend_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      alphaTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
            }
            else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** color_twist ********************/

RppStatus rppt_color_twist(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32f *brightnessTensor,
                           Rpp32f *contrastTensor,
                           Rpp32f *hueTensor,
                           Rpp32f *saturationTensor,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle,
                           RppBackend executionBackend)
{

    if (srcDescPtr->c != 3) return RPP_ERROR_INVALID_CHANNELS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            color_twist_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                          srcDescPtr,
                                          static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                          dstDescPtr,
                                          brightnessTensor,
                                          contrastTensor,
                                          hueTensor,
                                          saturationTensor,
                                          roiTensorPtrSrc,
                                          roiType,
                                          layoutParams,
                                          handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            color_twist_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            brightnessTensor,
                                            contrastTensor,
                                            hueTensor,
                                            saturationTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            color_twist_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            brightnessTensor,
                                            contrastTensor,
                                            hueTensor,
                                            saturationTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            color_twist_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                          srcDescPtr,
                                          static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                          dstDescPtr,
                                          brightnessTensor,
                                          contrastTensor,
                                          hueTensor,
                                          saturationTensor,
                                          roiTensorPtrSrc,
                                          roiType,
                                          layoutParams,
                                          handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_color_twist_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_color_twist_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_color_twist_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_color_twist_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** hue ********************/

RppStatus rppt_hue(RppPtr_t srcPtr,
                   RpptDescPtr srcDescPtr,
                   RppPtr_t dstPtr,
                   RpptDescPtr dstDescPtr,
                   Rpp32f *hueTensor,
                   RpptROIPtr roiTensorPtrSrc,
                   RpptRoiType roiType,
                   rppHandle_t rppHandle,
                   RppBackend executionBackend)
{
    if (srcDescPtr->c != 3) return RPP_ERROR_INVALID_CHANNELS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hue_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  hueTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hue_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    hueTensor,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hue_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    hueTensor,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hue_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  hueTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_hue_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                hueTensor,
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_hue_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                hueTensor,
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_hue_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                hueTensor,
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_hue_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                hueTensor,
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** saturation ********************/

RppStatus rppt_saturation(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *saturationTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->c != 3) return RPP_ERROR_INVALID_CHANNELS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            saturation_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            saturation_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           saturationTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            saturation_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           saturationTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            saturation_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_saturation_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_saturation_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_saturation_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_saturation_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}


/******************** color_jitter ********************/

RppStatus rppt_color_jitter(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *brightnessTensor,
                            Rpp32f *contrastTensor,
                            Rpp32f *hueTensor,
                            Rpp32f *saturationTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle,
                            RppBackend executionBackend)
{
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            color_jitter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           brightnessTensor,
                                           contrastTensor,
                                           hueTensor,
                                           saturationTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            color_jitter_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            brightnessTensor,
                                            contrastTensor,
                                            hueTensor,
                                            saturationTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            color_jitter_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            brightnessTensor,
                                            contrastTensor,
                                            hueTensor,
                                            saturationTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            color_jitter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           brightnessTensor,
                                           contrastTensor,
                                           hueTensor,
                                           saturationTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
        return RPP_ERROR_NOT_IMPLEMENTED;
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** color_cast ********************/

RppStatus rppt_color_cast(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptRGB *rgbTensor,
                          Rpp32f *alphaTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->c != 3) return RPP_ERROR_INVALID_CHANNELS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            color_cast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rgbTensor,
                                         alphaTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            color_cast_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           rgbTensor,
                                           alphaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            color_cast_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           rgbTensor,
                                           alphaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            color_cast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rgbTensor,
                                         alphaTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_color_cast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_color_cast_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_color_cast_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_color_cast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** exposure ********************/

RppStatus rppt_exposure(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32f *exposureFactorTensor,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle,
                        RppBackend executionBackend)
{
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            exposure_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       exposureFactorTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            exposure_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         exposureFactorTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            exposure_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         exposureFactorTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            exposure_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       exposureFactorTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_exposure_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_exposure_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_exposure_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_exposure_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}


/******************** contrast ********************/

RppStatus rppt_contrast(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32f *contrastFactorTensor,
                        Rpp32f *contrastCenterTensor,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle,
                        RppBackend executionBackend)
{
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            contrast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       contrastFactorTensor,
                                       contrastCenterTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            contrast_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         contrastFactorTensor,
                                         contrastCenterTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            contrast_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         contrastFactorTensor,
                                         contrastCenterTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            contrast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       contrastFactorTensor,
                                       contrastCenterTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_contrast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_contrast_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_contrast_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_contrast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** lut ********************/

RppStatus rppt_lut(RppPtr_t srcPtr,
                   RpptDescPtr srcDescPtr,
                   RppPtr_t dstPtr,
                   RpptDescPtr dstDescPtr,
                   RppPtr_t lutPtr,
                   RpptROIPtr roiTensorPtrSrc,
                   RpptRoiType roiType,
                   rppHandle_t rppHandle,
                   RppBackend executionBackend)
{
    if (srcDescPtr->dataType != RpptDataType::U8 && srcDescPtr->dataType != RpptDataType::I8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            lut_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  static_cast<Rpp8u*>(lutPtr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            lut_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   static_cast<Rpp16f*>(lutPtr),
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            lut_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   static_cast<Rpp32f*>(lutPtr),
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            lut_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  static_cast<Rpp8s*>(lutPtr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                static_cast<Rpp8u*>(lutPtr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                static_cast<half*>(lutPtr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                static_cast<Rpp32f*>(lutPtr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_lut_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                static_cast<Rpp8s*>(lutPtr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** color_temperature ********************/

RppStatus rppt_color_temperature(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32s *adjustmentValueTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rppHandle_t rppHandle,
                                 RppBackend executionBackend)
{
    if (srcDescPtr->c != 3) return RPP_ERROR_INVALID_CHANNELS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            color_temperature_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                adjustmentValueTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            color_temperature_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  adjustmentValueTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            color_temperature_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  adjustmentValueTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            color_temperature_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                adjustmentValueTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_color_temperature_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_color_temperature_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_color_temperature_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_color_temperature_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}


/******************** histogram_equalize ********************/

RppStatus rppt_histogram_equalize(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rppHandle_t rppHandle,
                                  RppBackend executionBackend)
{
    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            histogram_equalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                 dstDescPtr,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_histogram_equalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               roiTensorPtrSrc,
                                               roiType,
                                               handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

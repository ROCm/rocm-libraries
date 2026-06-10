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
#include "rppt_tensor_morphological_operations.h"
#include "host_tensor_executors.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif // GPU_SUPPORT

/******************** erode ********************/

RppStatus rppt_erode_host(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32u kernelSize,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        erode_char_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               kernelSize,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        erode_float_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        erode_float_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        erode_char_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               kernelSize,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    return RPP_SUCCESS;
}

/******************** dilate ********************/

RppStatus rppt_dilate_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32u kernelSize,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        dilate_char_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        dilate_float_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 kernelSize,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        dilate_float_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 kernelSize,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        dilate_char_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/


/******************** erode ********************/

RppStatus rppt_erode(RppPtr_t srcPtr,
                     RpptDescPtr srcDescPtr,
                     RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr,
                     Rpp32u kernelSize,
                     RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType,
                     rppHandle_t rppHandle,
                     RppBackend executionBackend)
{
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_erode_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_erode_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_erode_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_erode_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** dilate ********************/

RppStatus rppt_dilate(RppPtr_t srcPtr,
                      RpptDescPtr srcDescPtr,
                      RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr,
                      Rpp32u kernelSize,
                      RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType,
                      rppHandle_t rppHandle,
                      RppBackend executionBackend)
{
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_dilate_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_dilate_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_dilate_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_dilate_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                kernelSize,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

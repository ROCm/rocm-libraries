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
#include "rppi_validate.hpp"
#include "rppt_tensor_filter_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** box_filter ********************/

RppStatus rppt_box_filter_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32u kernelSize,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        box_filter_char_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
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
        box_filter_float_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
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
        box_filter_float_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
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
        box_filter_char_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    kernelSize,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** gaussian_filter ********************/

RppStatus rppt_gaussian_filter_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *stdDevTensor,
                                    Rpp32u kernelSize,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        gaussian_filter_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    stdDevTensor,
                                    kernelSize,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        gaussian_filter_host_tensor(reinterpret_cast<Rpp16f *>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    reinterpret_cast<Rpp16f *>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    stdDevTensor,
                                    kernelSize,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        gaussian_filter_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    stdDevTensor,
                                    kernelSize,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        gaussian_filter_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    stdDevTensor,
                                    kernelSize,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** box_filter ********************/

RppStatus rppt_box_filter_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32u kernelSize,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_box_filter_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
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
        hip_exec_box_filter_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_box_filter_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_box_filter_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** gaussian_filter ********************/

RppStatus rppt_gaussian_filter_gpu(RppPtr_t srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *stdDevTensor,
                                   Rpp32u kernelSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;

    Rpp32u paramIndex = 0;
    copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_gaussian_filter_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
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
        hip_exec_gaussian_filter_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
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
        hip_exec_gaussian_filter_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
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
        hip_exec_gaussian_filter_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        kernelSize,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT

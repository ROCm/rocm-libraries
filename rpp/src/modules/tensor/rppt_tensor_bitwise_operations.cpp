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
#include "rppt_tensor_bitwise_operations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** bitwise AND ********************/

RppStatus rppt_bitwise_and_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        bitwise_and_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** bitwise XOR ********************/

RppStatus rppt_bitwise_xor_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        bitwise_xor_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                      static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** bitwise OR ********************/

RppStatus rppt_bitwise_or_host(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        bitwise_or_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                     static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** bitwise NOT ********************/

RppStatus rppt_bitwise_not_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        bitwise_not_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
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

/******************** bitwise AND ********************/

RppStatus rppt_bitwise_and_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_bitwise_and_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** bitwise XOR ********************/

RppStatus rppt_bitwise_xor_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_bitwise_xor_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** bitwise OR ********************/

RppStatus rppt_bitwise_or_gpu(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_bitwise_or_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                   static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_bitwise_not_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_bitwise_not_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
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

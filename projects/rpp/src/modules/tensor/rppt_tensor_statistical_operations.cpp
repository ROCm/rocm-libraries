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
#include "rppt_tensor_statistical_operations.h"
#include "kernel_dims.hpp"
#include "host_tensor_executors.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif // GPU_SUPPORT

/******************** tensor_sum ********************/

RppStatus rppt_tensor_sum(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t tensorSumArr,
                          Rpp32u tensorSumArrLength,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->c == 1)
    {
        if (tensorSumArrLength < srcDescPtr->n)      // sum of single channel
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorSumArrLength < srcDescPtr->n * 4)  // sum of each channel, and total sum of all 3 channels
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            tensor_sum_u8_u64_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp64u*>(tensorSumArr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            tensor_sum_f16_f32_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(tensorSumArr),
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            tensor_sum_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(tensorSumArr),
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            tensor_sum_i8_i64_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp64s*>(tensorSumArr),
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hip_exec_tensor_sum(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp64u*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hip_exec_tensor_sum(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hip_exec_tensor_sum(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hip_exec_tensor_sum(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp64s*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_min ********************/

RppStatus rppt_tensor_min(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t minArr,
                          Rpp32u minArrLength,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->c == 1)
    {
        if (minArrLength < srcDescPtr->n)      // 1 min for each image
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (minArrLength < srcDescPtr->n * 4)  // min of each channel, and min of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            tensor_min_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(minArr),
                                  minArrLength,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            tensor_min_f16_f16_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp16f*>(minArr),
                                     minArrLength,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            tensor_min_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(minArr),
                                     minArrLength,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            tensor_min_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(minArr),
                                  minArrLength,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hip_exec_tensor_min(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(minArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hip_exec_tensor_min(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<half*>(minArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hip_exec_tensor_min(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(minArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hip_exec_tensor_min(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(minArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_max ********************/

RppStatus rppt_tensor_max(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t maxArr,
                          Rpp32u maxArrLength,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (srcDescPtr->c == 1)
    {
        if (maxArrLength < srcDescPtr->n)      // 1 max for each image
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (maxArrLength < srcDescPtr->n * 4)  // max of each channel, and max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            tensor_max_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(maxArr),
                                  maxArrLength,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            tensor_max_f16_f16_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp16f*>(maxArr),
                                     maxArrLength,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            tensor_max_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(maxArr),
                                     maxArrLength,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            tensor_max_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(maxArr),
                                  maxArrLength,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hip_exec_tensor_max(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(maxArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hip_exec_tensor_max(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<half*>(maxArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hip_exec_tensor_max(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(maxArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hip_exec_tensor_max(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(maxArr),
                                roiTensorPtrSrc,
                                roiType,
                                handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** normalize_ND ********************/

RppStatus rppt_normalize(RppPtr_t srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32u axisMask,
                         Rpp32f *meanTensor,
                         Rpp32f *stdDevTensor,
                         Rpp8u computeMeanStddev,
                         Rpp32f scale,
                         Rpp32f shift,
                         Rpp32u *roiTensor,
                         rppHandle_t rppHandle,
                         RppBackend executionBackend)
{
    if (srcGenericDescPtr->dataType != dstGenericDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams;
        Rpp32u tensorDim = srcGenericDescPtr->numDims - 1;
        if (tensorDim == 3 && (srcGenericDescPtr->layout == RpptLayout::NHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[3]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);
        else if(tensorDim == 2 && (srcGenericDescPtr->layout == RpptLayout::NHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[2]);

        if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
        {
            normalize_generic_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                          srcGenericDescPtr,
                                          static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                          dstGenericDescPtr,
                                          axisMask,
                                          meanTensor,
                                          stdDevTensor,
                                          computeMeanStddev,
                                          scale,
                                          shift,
                                          roiTensor,
                                          layoutParams,
                                          handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
        {
            normalize_generic_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                          srcGenericDescPtr,
                                          reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                          dstGenericDescPtr,
                                          axisMask,
                                          meanTensor,
                                          stdDevTensor,
                                          computeMeanStddev,
                                          scale,
                                          shift,
                                          roiTensor,
                                          layoutParams,
                                          handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
        {
            normalize_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                          srcGenericDescPtr,
                                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                          dstGenericDescPtr,
                                          axisMask,
                                          meanTensor,
                                          stdDevTensor,
                                          computeMeanStddev,
                                          scale,
                                          shift,
                                          roiTensor,
                                          layoutParams,
                                          handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
        {
            normalize_generic_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                          srcGenericDescPtr,
                                          static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                          dstGenericDescPtr,
                                          axisMask,
                                          meanTensor,
                                          stdDevTensor,
                                          computeMeanStddev,
                                          scale,
                                          shift,
                                          roiTensor,
                                          layoutParams,
                                          handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                      srcGenericDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                      dstGenericDescPtr,
                                      axisMask,
                                      meanTensor,
                                      stdDevTensor,
                                      computeMeanStddev,
                                      scale,
                                      shift,
                                      roiTensor,
                                      handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_normalize_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr,
                                      axisMask,
                                      meanTensor,
                                      stdDevTensor,
                                      computeMeanStddev,
                                      scale,
                                      shift,
                                      roiTensor,
                                      handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_normalize_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr,
                                      axisMask,
                                      meanTensor,
                                      stdDevTensor,
                                      computeMeanStddev,
                                      scale,
                                      shift,
                                      roiTensor,
                                      handle);
        }
        else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                      srcGenericDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                      dstGenericDescPtr,
                                      axisMask,
                                      meanTensor,
                                      stdDevTensor,
                                      computeMeanStddev,
                                      scale,
                                      shift,
                                      roiTensor,
                                      handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_mean ********************/

RppStatus rppt_tensor_mean(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t tensorMeanArr,
                           Rpp32u tensorMeanArrLength,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle,
                           RppBackend executionBackend)
{
    if ((srcDescPtr->c == 1 && tensorMeanArrLength < srcDescPtr->n) ||        // Mean of single channel
        (srcDescPtr->c == 3 && tensorMeanArrLength < srcDescPtr->n * 4))      // Mean of each channel, and total Mean of all 3 channels / image
        return RPP_ERROR_NOT_ENOUGH_MEMORY;
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH)|| (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            tensor_mean_u8_f32_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(tensorMeanArr),
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            tensor_mean_f16_f32_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorMeanArr),
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            tensor_mean_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorMeanArr),
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            tensor_mean_i8_f32_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(tensorMeanArr),
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hip_exec_tensor_mean<Rpp8u, Rpp32u>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp32f*>(tensorMeanArr),
                                                roiTensorPtrSrc,
                                                roiType,
                                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hip_exec_tensor_mean<half, float>(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                static_cast<Rpp32f*>(tensorMeanArr),
                                                roiTensorPtrSrc,
                                                roiType,
                                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hip_exec_tensor_mean<Rpp32f, float>(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                static_cast<Rpp32f*>(tensorMeanArr),
                                                roiTensorPtrSrc,
                                                roiType,
                                                handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hip_exec_tensor_mean<Rpp8s, Rpp32s>(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp32f*>(tensorMeanArr),
                                                roiTensorPtrSrc,
                                                roiType,
                                                handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor_stddev ********************/

RppStatus rppt_tensor_stddev(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t tensorStddevArr,
                             Rpp32u tensorStddevArrLength,
                             Rpp32f *meanTensor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle,
                             RppBackend executionBackend)
{
    if ((srcDescPtr->c == 1 && tensorStddevArrLength < srcDescPtr->n) ||        // Stddev of single channel
        (srcDescPtr->c == 3 && tensorStddevArrLength < srcDescPtr->n * 4))      // Stddev of each channel, and total Stddev of all 3 channels / image
        return RPP_ERROR_NOT_ENOUGH_MEMORY;
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            tensor_stddev_u8_f32_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(tensorStddevArr),
                                      meanTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            tensor_stddev_f16_f32_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            tensor_stddev_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            tensor_stddev_i8_f32_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(tensorStddevArr),
                                      meanTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hip_exec_tensor_stddev(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp32f*>(tensorStddevArr),
                                   meanTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hip_exec_tensor_stddev(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   static_cast<Rpp32f*>(tensorStddevArr),
                                   meanTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hip_exec_tensor_stddev(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   static_cast<Rpp32f*>(tensorStddevArr),
                                   meanTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   handle);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hip_exec_tensor_stddev(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp32f*>(tensorStddevArr),
                                   meanTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** threshold ********************/

RppStatus rppt_threshold(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32f *minTensor,
                         Rpp32f *maxTensor,
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
            threshold_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        minTensor,
                                        maxTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
           threshold_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         minTensor,
                                         maxTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            threshold_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          minTensor,
                                          maxTensor,
                                          roiTensorPtrSrc,
                                          roiType,
                                          layoutParams,
                                          handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            threshold_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        minTensor,
                                        maxTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_threshold_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      minTensor,
                                      maxTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
           hip_exec_threshold_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     minTensor,
                                     maxTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_threshold_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      minTensor,
                                      maxTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_threshold_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      minTensor,
                                      maxTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

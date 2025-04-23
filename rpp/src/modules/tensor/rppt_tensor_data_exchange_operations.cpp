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
#include "rppt_tensor_data_exchange_operations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** copy ********************/

RppStatus rppt_copy_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        copy_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        copy_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        copy_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        copy_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** swap_channels ********************/

RppStatus rppt_swap_channels_host(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        swap_channels_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        swap_channels_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          layoutParams,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        swap_channels_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          layoutParams,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        swap_channels_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_to_greyscale ********************/

RppStatus rppt_color_to_greyscale_host(RppPtr_t srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptSubpixelLayout srcSubpixelLayout,
                                       rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
        return RPP_ERROR_INVALID_SRC_CHANNELS;
    if (dstDescPtr->c != 1)
        return RPP_ERROR_INVALID_DST_CHANNELS;
    if (dstDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_DST_LAYOUT;

    Rpp32f channelWeights[3];
    if (srcSubpixelLayout == RpptSubpixelLayout::RGBtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_RED;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_BLUE;
    }
    else if (srcSubpixelLayout == RpptSubpixelLayout::BGRtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_BLUE;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_RED;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_to_greyscale_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             channelWeights,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_to_greyscale_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               layoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_to_greyscale_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               layoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_to_greyscale_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             channelWeights,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** copy ********************/

RppStatus rppt_copy_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_copy_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_copy_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_copy_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_copy_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** swap_channels ********************/

RppStatus rppt_swap_channels_gpu(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_swap_channels_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_swap_channels_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_swap_channels_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_swap_channels_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** color_to_greyscale ********************/

RppStatus rppt_color_to_greyscale_gpu(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptSubpixelLayout srcSubpixelLayout,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->c != 3)
        return RPP_ERROR_INVALID_SRC_CHANNELS;
    if (dstDescPtr->c != 1)
        return RPP_ERROR_INVALID_DST_CHANNELS;
    if (dstDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_DST_LAYOUT;

    Rpp32f channelWeights[3];
    if (srcSubpixelLayout == RpptSubpixelLayout::RGBtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_RED;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_BLUE;
    }
    else if (srcSubpixelLayout == RpptSubpixelLayout::BGRtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_BLUE;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_RED;
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_to_greyscale_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_to_greyscale_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_to_greyscale_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_to_greyscale_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT

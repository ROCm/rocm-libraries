/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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
#include "rppt_tensor_data_exchange_operations.h"
#include "host_tensor_executors.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif // GPU_SUPPORT

/******************** copy ********************/

RppStatus rppt_copy(RppPtr_t srcPtr,
                    RpptDescPtr srcDescPtr,
                    RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr,
                    rppHandle_t rppHandle,
                    RppBackend executionBackend)
{
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            copy_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   layoutParams,
                                   handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            copy_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     layoutParams,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            copy_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     layoutParams,
                                     handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            copy_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   layoutParams,
                                   handle);
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_copy_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_copy_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_copy_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_copy_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 handle);
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** channel_permute ********************/

RppStatus rppt_channel_permute(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32u *permutationTensor,
                               rppHandle_t rppHandle,
                               RppBackend executionBackend)
{

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            channel_permute_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              permutationTensor,
                                              layoutParams,
                                              handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            channel_permute_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                permutationTensor,
                                                layoutParams,
                                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            channel_permute_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                permutationTensor,
                                                layoutParams,
                                                handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            channel_permute_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              permutationTensor,
                                              layoutParams,
                                              handle);
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_channel_permute_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            permutationTensor,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_channel_permute_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            permutationTensor,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_channel_permute_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                            srcDescPtr,
                                            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                            dstDescPtr,
                                            permutationTensor,
                                            handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_channel_permute_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            permutationTensor,
                                            handle);
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** color_to_greyscale ********************/

RppStatus rppt_color_to_greyscale(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptSubpixelLayout srcSubpixelLayout,
                                  rppHandle_t rppHandle,
                                  RppBackend executionBackend)
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

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            color_to_greyscale_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                 dstDescPtr,
                                                 channelWeights,
                                                 layoutParams,
                                                 handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            color_to_greyscale_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                   srcDescPtr,
                                                   (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   channelWeights,
                                                   layoutParams,
                                                   handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            color_to_greyscale_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                   srcDescPtr,
                                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   channelWeights,
                                                   layoutParams,
                                                   handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            color_to_greyscale_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                 dstDescPtr,
                                                 channelWeights,
                                                 layoutParams,
                                                 handle);
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            hip_exec_color_to_greyscale_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               channelWeights,
                                               handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            hip_exec_color_to_greyscale_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_color_to_greyscale_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               handle);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            hip_exec_color_to_greyscale_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               channelWeights,
                                               handle);
        }

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** yuv_to_rgb ********************/

RppStatus rppt_yuv_to_rgb(RppPtr_t srcYPtr,
                          RppPtr_t srcUVPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32u src_y_pitch,
                          Rpp32u src_uv_pitch,
                          Rpp32u dst_pitch,
                          Rpp32u width,
                          Rpp32u height,
                          RpptColorStandard col_standard,
                          RpptColorRange color_range,
                          rppHandle_t rppHandle,
                          RppBackend executionBackend)
{
    if (executionBackend != RppBackend::RPP_HIP_BACKEND)
        return RPP_ERROR_INCOMPATIBLE_BACKEND;
    if (srcDescPtr->dataType != RpptDataType::U8 || dstDescPtr->dataType != RpptDataType::U8)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

#ifdef GPU_SUPPORT
    if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        return hip_exec_yuv_to_rgb<Rpp8u>(static_cast<Rpp8u*>(srcYPtr),
                                        src_y_pitch,
                                        static_cast<Rpp8u*>(srcUVPtr),
                                        src_uv_pitch,
                                        static_cast<Rpp8u*>(dstPtr),
                                        dst_pitch,
                                        width,
                                        height,
                                        col_standard,
                                        color_range,
                                        handle);
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** yuv_to_rgb_cubic_v ********************/

RppStatus rppt_yuv_to_rgb_cubic_v(RppPtr_t srcYPtr,
                                    RppPtr_t srcUVPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u src_y_pitch,
                                    Rpp32u src_uv_pitch,
                                    Rpp32u dst_pitch,
                                    Rpp32u width,
                                    Rpp32u height,
                                    RpptColorStandard col_standard,
                                    RpptColorRange color_range,
                                    rppHandle_t rppHandle,
                                    RppBackend executionBackend)
{
    if (executionBackend != RppBackend::RPP_HIP_BACKEND)
        return RPP_ERROR_INCOMPATIBLE_BACKEND;
    if (srcDescPtr->dataType != RpptDataType::U8 || dstDescPtr->dataType != RpptDataType::U8)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (height < 2 || width < 2)
        return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

#ifdef GPU_SUPPORT
    if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        return hip_exec_yuv_to_rgb_cubic_v<Rpp8u>(static_cast<Rpp8u*>(srcYPtr),
                                                    src_y_pitch,
                                                    static_cast<Rpp8u*>(srcUVPtr),
                                                    src_uv_pitch,
                                                    static_cast<Rpp8u*>(dstPtr),
                                                    dst_pitch,
                                                    width,
                                                    height,
                                                    col_standard,
                                                    color_range,
                                                    handle);
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** yuv_to_rgb_linear_v ********************/

RppStatus rppt_yuv_to_rgb_linear_v(RppPtr_t srcYPtr,
                                     RppPtr_t srcUVPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u src_y_pitch,
                                     Rpp32u src_uv_pitch,
                                     Rpp32u dst_pitch,
                                     Rpp32u width,
                                     Rpp32u height,
                                     RpptColorStandard col_standard,
                                     RpptColorRange color_range,
                                     rppHandle_t rppHandle,
                                     RppBackend executionBackend)
{
    if (executionBackend != RppBackend::RPP_HIP_BACKEND)
        return RPP_ERROR_INCOMPATIBLE_BACKEND;
    if (srcDescPtr->dataType != RpptDataType::U8 || dstDescPtr->dataType != RpptDataType::U8)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (height < 2 || width < 2)
        return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

#ifdef GPU_SUPPORT
    if ((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        return hip_exec_yuv_to_rgb_linear_v<Rpp8u>(static_cast<Rpp8u*>(srcYPtr),
                                                     src_y_pitch,
                                                     static_cast<Rpp8u*>(srcUVPtr),
                                                     src_uv_pitch,
                                                     static_cast<Rpp8u*>(dstPtr),
                                                     dst_pitch,
                                                     width,
                                                     height,
                                                     col_standard,
                                                     color_range,
                                                     handle);
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

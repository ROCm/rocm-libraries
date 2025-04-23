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
#include "rppi_fused_functions.h"
#include "host_fused_functions.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** color_twist ********************/

RppStatus color_twist_host_helper(RppiChnFormat chn_format,
                                  Rpp32u num_of_channels,
                                  RPPTensorDataType tensor_type,
                                  RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *alpha,
                                  Rpp32f *beta,
                                  Rpp32f *hueShift,
                                  Rpp32f *saturationFactor,
                                  Rpp32u outputFormatToggle,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        color_twist_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      alpha,
                                      beta,
                                      hueShift,
                                      saturationFactor,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      outputFormatToggle,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      chn_format,
                                      num_of_channels,
                                      rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        color_twist_f16_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp16f *>(dstPtr),
                                           alpha,
                                           beta,
                                           hueShift,
                                           saturationFactor,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        color_twist_f32_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp32f *>(dstPtr),
                                           alpha,
                                           beta,
                                           hueShift,
                                           saturationFactor,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        color_twist_i8_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         static_cast<Rpp8s *>(dstPtr),
                                         alpha,
                                         beta,
                                         hueShift,
                                         saturationFactor,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         outputFormatToggle,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         chn_format,
                                         num_of_channels,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** crop ********************/

RppStatus crop_host_helper(RppiChnFormat chn_format,
                           Rpp32u num_of_channels,
                           RPPTensorDataType tensorInType,
                           RPPTensorDataType tensorOutType,
                           RppPtr_t srcPtr,
                           RppiSize *srcSize,
                           RppiSize maxSrcSize,
                           RppPtr_t dstPtr,
                           RppiSize *dstSize,
                           RppiSize maxDstSize,
                           Rpp32u *crop_pos_x,
                           Rpp32u *crop_pos_y,
                           Rpp32u outputFormatToggle,
                           Rpp32u nbatchSize,
                           rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensorInType == RPPTensorDataType::U8)
    {
        if (tensorOutType == RPPTensorDataType::U8)
        {
            crop_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u *>(dstPtr),
                                   dstSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                   crop_pos_x,
                                   crop_pos_y,
                                   outputFormatToggle,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   chn_format,
                                   num_of_channels,
                                   rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP16)
        {
            crop_host_u_f_batch<Rpp8u, Rpp16f>(static_cast<Rpp8u *>(srcPtr),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp16f *>(dstPtr),
                                               dstSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                               crop_pos_x,
                                               crop_pos_y,
                                               outputFormatToggle,
                                               rpp::deref(rppHandle).GetBatchSize(),
                                               chn_format,
                                               num_of_channels,
                                               rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP32)
        {
            crop_host_u_f_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u *>(srcPtr),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp32f *>(dstPtr),
                                               dstSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                               crop_pos_x,
                                               crop_pos_y,
                                               outputFormatToggle,
                                               rpp::deref(rppHandle).GetBatchSize(),
                                               chn_format,
                                               num_of_channels,
                                               rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::I8)
        {
            crop_host_u_i_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u *>(srcPtr),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp8s *>(dstPtr),
                                              dstSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                              crop_pos_x,
                                              crop_pos_y,
                                              outputFormatToggle,
                                              rpp::deref(rppHandle).GetBatchSize(),
                                              chn_format,
                                              num_of_channels,
                                              rpp::deref(rppHandle));
        }
    }
    else if (tensorInType == RPPTensorDataType::FP16)
    {
        crop_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                srcSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                static_cast<Rpp16f *>(dstPtr),
                                dstSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                crop_pos_x,
                                crop_pos_y,
                                outputFormatToggle,
                                rpp::deref(rppHandle).GetBatchSize(),
                                chn_format,
                                num_of_channels,
                                rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::FP32)
    {
        crop_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                srcSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                static_cast<Rpp32f *>(dstPtr),
                                dstSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                crop_pos_x,
                                crop_pos_y,
                                outputFormatToggle,
                                rpp::deref(rppHandle).GetBatchSize(),
                                chn_format,
                                num_of_channels,
                                rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::I8)
    {
        crop_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8s *>(dstPtr),
                               dstSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                               crop_pos_x,
                               crop_pos_y,
                               outputFormatToggle,
                               rpp::deref(rppHandle).GetBatchSize(),
                               chn_format,
                               num_of_channels,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** crop_mirror_normalize ********************/

RppStatus crop_mirror_normalize_host_helper(RppiChnFormat chn_format,
                                            Rpp32u num_of_channels,
                                            RPPTensorDataType tensorInType,
                                            RPPTensorDataType tensorOutType,
                                            RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            RppiSize *dstSize,
                                            RppiSize maxDstSize,
                                            Rpp32u *crop_pos_x,
                                            Rpp32u *crop_pos_y,
                                            Rpp32f *mean,
                                            Rpp32f *stdDev,
                                            Rpp32u *mirrorFlag,
                                            Rpp32u outputFormatToggle,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensorInType == RPPTensorDataType::U8)
    {
        if (tensorOutType == RPPTensorDataType::U8)
        {
            crop_mirror_normalize_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                                    srcSize,
                                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                    static_cast<Rpp8u *>(dstPtr),
                                                    dstSize,
                                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                    crop_pos_x,
                                                    crop_pos_y,
                                                    mean,
                                                    stdDev,
                                                    mirrorFlag,
                                                    outputFormatToggle,
                                                    rpp::deref(rppHandle).GetBatchSize(),
                                                    chn_format,
                                                    num_of_channels,
                                                    rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP16)
        {
            crop_mirror_normalize_u8_f_host_batch<Rpp8u, Rpp16f>(static_cast<Rpp8u *>(srcPtr),
                                                                 srcSize,
                                                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                                 static_cast<Rpp16f *>(dstPtr),
                                                                 dstSize,
                                                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                                 crop_pos_x,
                                                                 crop_pos_y,
                                                                 mean,
                                                                 stdDev,
                                                                 mirrorFlag,
                                                                 outputFormatToggle,
                                                                 rpp::deref(rppHandle).GetBatchSize(),
                                                                 chn_format,
                                                                 num_of_channels,
                                                                 rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP32)
        {
            crop_mirror_normalize_u8_f_host_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u *>(srcPtr),
                                                                 srcSize,
                                                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                                 static_cast<Rpp32f *>(dstPtr),
                                                                 dstSize,
                                                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                                 crop_pos_x,
                                                                 crop_pos_y,
                                                                 mean,
                                                                 stdDev,
                                                                 mirrorFlag,
                                                                 outputFormatToggle,
                                                                 rpp::deref(rppHandle).GetBatchSize(),
                                                                 chn_format,
                                                                 num_of_channels,
                                                                 rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::I8)
        {
            crop_mirror_normalize_u8_i8_host_batch(static_cast<Rpp8u *>(srcPtr),
                                                   srcSize,
                                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                   static_cast<Rpp8s *>(dstPtr),
                                                   dstSize,
                                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                   crop_pos_x,
                                                   crop_pos_y,
                                                   mean,
                                                   stdDev,
                                                   mirrorFlag,
                                                   outputFormatToggle,
                                                   rpp::deref(rppHandle).GetBatchSize(),
                                                   chn_format,
                                                   num_of_channels,
                                                   rpp::deref(rppHandle));
        }
    }
    else if (tensorInType == RPPTensorDataType::FP16)
    {
        crop_mirror_normalize_f16_host_batch(static_cast<Rpp16f *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp16f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             crop_pos_x,
                                             crop_pos_y,
                                             mean,
                                             stdDev,
                                             mirrorFlag,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::FP32)
    {
        crop_mirror_normalize_f32_host_batch(static_cast<Rpp32f *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp32f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             crop_pos_x,
                                             crop_pos_y,
                                             mean,
                                             stdDev,
                                             mirrorFlag,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::I8)
    {
        crop_mirror_normalize_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp8s *>(dstPtr),
                                                dstSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                crop_pos_x,
                                                crop_pos_y,
                                                mean,
                                                stdDev,
                                                mirrorFlag,
                                                outputFormatToggle,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                chn_format,
                                                num_of_channels,
                                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** resize_crop_mirror ********************/

RppStatus resize_crop_mirror_host_helper(RppiChnFormat chn_format,
                                         Rpp32u num_of_channels,
                                         RPPTensorDataType tensor_type,
                                         RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         RppiSize *dstSize,
                                         RppiSize maxDstSize,
                                         Rpp32u *xRoiBegin,
                                         Rpp32u *xRoiEnd,
                                         Rpp32u *yRoiBegin,
                                         Rpp32u *yRoiEnd,
                                         Rpp32u *mirrorFlag,
                                         Rpp32u outputFormatToggle,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        resize_crop_mirror_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             xRoiBegin,
                                             xRoiEnd,
                                             yRoiBegin,
                                             yRoiEnd,
                                             mirrorFlag,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        resize_crop_mirror_f16_host_batch(static_cast<Rpp16f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp16f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          xRoiBegin,
                                          xRoiEnd,
                                          yRoiBegin,
                                          yRoiEnd,
                                          mirrorFlag,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        resize_crop_mirror_f32_host_batch(static_cast<Rpp32f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp32f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          xRoiBegin,
                                          xRoiEnd,
                                          yRoiBegin,
                                          yRoiEnd,
                                          mirrorFlag,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        resize_crop_mirror_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8s *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             xRoiBegin,
                                             xRoiEnd,
                                             yRoiBegin,
                                             yRoiEnd,
                                             mirrorFlag,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_mirror_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag,Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** resize_mirror_normalize ********************/

RppStatus resize_mirror_normalize_host_helper(RppiChnFormat chn_format,
                                              Rpp32u num_of_channels,
                                              RPPTensorDataType tensor_type,
                                              RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              RppiSize *dstSize,
                                              RppiSize maxDstSize,
                                              Rpp32f *batch_mean,
                                              Rpp32f *batch_stdDev,
                                              Rpp32u *mirrorFlag,
                                              Rpp32u outputFormatToggle,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        resize_mirror_normalize_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                                  srcSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                  static_cast<Rpp8u *>(dstPtr),
                                                  dstSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                  batch_mean,
                                                  batch_stdDev,
                                                  mirrorFlag,
                                                  outputFormatToggle,
                                                  rpp::deref(rppHandle).GetBatchSize(),
                                                  chn_format,
                                                  num_of_channels,
                                                  rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, batch_mean, batch_stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag,Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, batch_mean, batch_stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, batch_mean, batch_stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** color_twist ********************/

RppStatus color_twist_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType in_tensor_type,
                             RPPTensorDataType out_tensor_type,
                             Rpp8u outputFormatToggle,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32f *alpha,
                             Rpp32f *beta,
                             Rpp32f *hueShift,
                             Rpp32f *saturationFactor,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiSize maxDstSize = maxSrcSize;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        color_twist_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            color_twist_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                         static_cast<Rpp8u *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            color_twist_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            color_twist_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            color_twist_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, nbatchSize, rppHandle));
}

/******************** crop ********************/

RppStatus crop_helper(RppiChnFormat chn_format,
                      Rpp32u num_of_channels,
                      RPPTensorDataType in_tensor_type,
                      RPPTensorDataType out_tensor_type,
                      RppPtr_t srcPtr,
                      RppiSize *srcSize,
                      RppiSize maxSrcSize,
                      RppPtr_t dstPtr,
                      RppiSize *dstSize,
                      RppiSize maxDstSize,
                      Rpp32u *crop_pos_x,
                      Rpp32u *crop_pos_y,
                      Rpp32u outputFormatToggle,
                      Rpp32u nbatchSize,
                      rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    Rpp32u paramIndex = 0;
    bool is_padded = true;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(crop_pos_x, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(crop_pos_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        crop_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            if (out_tensor_type == RPPTensorDataType::U8)
            {
                crop_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                      static_cast<Rpp8u *>(dstPtr),
                                      rpp::deref(rppHandle),
                                      tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                crop_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                crop_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                crop_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            crop_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                       static_cast<Rpp16f *>(dstPtr),
                                       rpp::deref(rppHandle),
                                       tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            crop_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                       static_cast<Rpp32f *>(dstPtr),
                                       rpp::deref(rppHandle),
                                       tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            crop_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                       static_cast<Rpp8s *>(dstPtr),
                                       rpp::deref(rppHandle),
                                       tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle, nbatchSize, rppHandle));
}

/******************** crop_mirror_normalize ********************/

RppStatus crop_mirror_normalize_helper(RppiChnFormat chn_format,
                                       Rpp32u num_of_channels,
                                       RPPTensorDataType in_tensor_type,
                                       RPPTensorDataType out_tensor_type,
                                       RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       RppiSize *dstSize,
                                       RppiSize maxDstSize,
                                       Rpp32u *crop_pos_x,
                                       Rpp32u *crop_pos_y,
                                       Rpp32f *mean,
                                       Rpp32f *std_dev,
                                       Rpp32u *mirrorFlag,
                                       Rpp32u outputFormatToggle,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    bool is_padded = true;

    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);

    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(crop_pos_x, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(crop_pos_y, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(mean, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(std_dev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(mirrorFlag, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        crop_mirror_normalize_cl_batch(static_cast<cl_mem>(srcPtr),
                                       static_cast<cl_mem>(dstPtr),
                                       rpp::deref(rppHandle),
                                       tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            if (out_tensor_type == RPPTensorDataType::U8)
            {
                crop_mirror_normalize_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                                       static_cast<Rpp8u *>(dstPtr),
                                                       rpp::deref(rppHandle),
                                                       tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                crop_mirror_normalize_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                                               static_cast<Rpp16f *>(dstPtr),
                                                               rpp::deref(rppHandle),
                                                               tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                crop_mirror_normalize_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                                               static_cast<Rpp32f *>(dstPtr),
                                                               rpp::deref(rppHandle),
                                                               tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                crop_mirror_normalize_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                                               static_cast<Rpp8s *>(dstPtr),
                                                               rpp::deref(rppHandle),
                                                               tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            crop_mirror_normalize_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                                        static_cast<Rpp16f *>(dstPtr),
                                                        rpp::deref(rppHandle),
                                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            crop_mirror_normalize_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                                        static_cast<Rpp32f *>(dstPtr),
                                                        rpp::deref(rppHandle),
                                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            crop_mirror_normalize_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                                        static_cast<Rpp8s *>(dstPtr),
                                                        rpp::deref(rppHandle),
                                                        tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, std_dev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** resize_crop_mirror ********************/

RppStatus resize_crop_mirror_helper(RppiChnFormat chn_format,
                                    Rpp32u num_of_channels,
                                    RPPTensorDataType in_tensor_type,
                                    RPPTensorDataType out_tensor_type,
                                    Rpp32u outputFormatToggle,
                                    RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    RppiSize *dstSize,
                                    RppiSize maxDstSize,
                                    Rpp32u *xRoiBegin,
                                    Rpp32u *xRoiEnd,
                                    Rpp32u *yRoiBegin,
                                    Rpp32u *yRoiEnd,
                                    Rpp32u *mirrorFlag,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(mirrorFlag, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        resize_crop_mirror_cl_batch(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            resize_crop_mirror_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp8u *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            resize_crop_mirror_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                                     static_cast<Rpp16f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            resize_crop_mirror_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                                     static_cast<Rpp32f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            resize_crop_mirror_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                                     static_cast<Rpp8s *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, yRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, yRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

#endif // GPU_SUPPORT

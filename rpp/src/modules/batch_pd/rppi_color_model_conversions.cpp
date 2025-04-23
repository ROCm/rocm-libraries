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
#include "rppi_color_model_conversions.h"
#include "host_color_model_conversions.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** hue ********************/

RppStatus
rppi_hueRGB_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp32f *hueShift,
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

    hueRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             hueShift,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PLANAR,
                             3,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp32f *hueShift,
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

    hueRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             hueShift,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PACKED,
                             3,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** saturation ********************/

RppStatus
rppi_saturationRGB_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *saturationFactor,
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

    saturationRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    saturationFactor,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *saturationFactor,
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

    saturationRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    saturationFactor,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** color_convert ********************/

RppStatus
rppi_color_convert_u8_pln3_batchPS_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        RppiColorConvertMode convert_mode,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if(convert_mode == RppiColorConvertMode::RGB_HSV)
    {
        color_convert_rgb_to_hsv_host_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr),
                                                           srcSize,
                                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                           static_cast<Rpp32f*>(dstPtr),
                                                           convert_mode,
                                                           rpp::deref(rppHandle).GetBatchSize(),
                                                           RPPI_CHN_PLANAR,
                                                           3,
                                                           rpp::deref(rppHandle));
    }
    else if(convert_mode == RppiColorConvertMode::HSV_RGB)
    {
        color_convert_hsv_to_rgb_host_batch<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr),
                                                           srcSize,
                                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                           static_cast<Rpp8u*>(dstPtr),
                                                           convert_mode,
                                                           rpp::deref(rppHandle).GetBatchSize(),
                                                           RPPI_CHN_PLANAR,
                                                           3,
                                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_color_convert_u8_pkd3_batchPS_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        RppiColorConvertMode convert_mode,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if(convert_mode == RppiColorConvertMode::RGB_HSV)
    {
        color_convert_rgb_to_hsv_host_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr),
                                                           srcSize,
                                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                           static_cast<Rpp32f*>(dstPtr),
                                                           convert_mode,
                                                           rpp::deref(rppHandle).GetBatchSize(),
                                                           RPPI_CHN_PACKED,
                                                           3,
                                                           rpp::deref(rppHandle));
    }
    else if(convert_mode == RppiColorConvertMode::HSV_RGB)
    {
        color_convert_hsv_to_rgb_host_batch<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr),
                                                           srcSize,
                                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                           static_cast<Rpp8u*>(dstPtr),
                                                           convert_mode,
                                                           rpp::deref(rppHandle).GetBatchSize(),
                                                           RPPI_CHN_PACKED,
                                                           3,
                                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_temperature ********************/

RppStatus
rppi_color_temperature_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32s *adjustmentValue,
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

    color_temperature_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8u*>(dstPtr),
                                        adjustmentValue,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                        rpp::deref(rppHandle).GetBatchSize(),
                                        RPPI_CHN_PLANAR,
                                        1,
                                        rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32s *adjustmentValue,
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

    color_temperature_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8u*>(dstPtr),
                                        adjustmentValue,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                        rpp::deref(rppHandle).GetBatchSize(),
                                        RPPI_CHN_PLANAR,
                                        3,
                                        rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32s *adjustmentValue,
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

    color_temperature_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8u*>(dstPtr),
                                        adjustmentValue,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                        rpp::deref(rppHandle).GetBatchSize(),
                                        RPPI_CHN_PACKED,
                                        3,
                                        rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** vignette ********************/

RppStatus
rppi_vignette_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *stdDev,
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

    vignette_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               stdDev,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *stdDev,
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

    vignette_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               stdDev,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *stdDev,
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

    vignette_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               stdDev,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** channel_extract ********************/

RppStatus
rppi_channel_extract_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *extractChannelNumber,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_extract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      extractChannelNumber,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      1,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_extract_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *extractChannelNumber,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_extract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      extractChannelNumber,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_extract_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *extractChannelNumber,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_extract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      extractChannelNumber,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PACKED,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** channel_combine ********************/

RppStatus
rppi_channel_combine_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                          RppPtr_t srcPtr2,
                                          RppPtr_t srcPtr3,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_combine_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(srcPtr3),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      1,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_combine_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                          RppPtr_t srcPtr2,
                                          RppPtr_t srcPtr3,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_combine_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(srcPtr3),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_combine_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                          RppPtr_t srcPtr2,
                                          RppPtr_t srcPtr3,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    channel_combine_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(srcPtr3),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PACKED,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** look_up_table ********************/

RppStatus
rppi_look_up_table_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp8u* lutPtr,
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

    look_up_table_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    static_cast<Rpp8u *>(lutPtr),
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    1,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_look_up_table_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp8u* lutPtr,
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

    look_up_table_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    static_cast<Rpp8u *>(lutPtr),
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_look_up_table_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp8u* lutPtr,
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

    look_up_table_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    static_cast<Rpp8u *>(lutPtr),
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** tensor_table_lookup ********************/

RppStatus
rppi_tensor_look_up_table_u8_host(RppPtr_t srcPtr,
                                  RppPtr_t dstPtr,
                                  RppPtr_t lutPtr,
                                  Rpp32u tensorDimension,
                                  RppPtr_t tensorDimensionValues)
{
    tensor_look_up_table_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                     static_cast<Rpp8u*>(dstPtr),
                                     static_cast<Rpp8u*>(lutPtr),
                                     tensorDimension,
                                     static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** hue ********************/

RppStatus
rppi_hueRGB_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *hueShift,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        hueRGB_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#elif defined(HIP_COMPILE)
    {
        hueRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *hueShift,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        hueRGB_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#elif defined(HIP_COMPILE)
    {
        hueRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PACKED,
                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** saturation ********************/

RppStatus
rppi_saturationRGB_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32f *saturationFactor,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        saturationRGB_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        saturationRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32f *saturationFactor,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        saturationRGB_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        saturationRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PACKED,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** color_convert ********************/

RppStatus
rppi_color_convert_u8_pln3_batchPS_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       RppiColorConvertMode convert_mode,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        color_convert_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               convert_mode,
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        if(convert_mode == RGB_HSV)
            color_convert_hip_batch_u8_fp32(static_cast<Rpp8u*>(srcPtr),
                                            static_cast<Rpp32f*>(dstPtr),
                                            RPPI_CHN_PLANAR,
                                            3,
                                            rpp::deref(rppHandle));
        else if(convert_mode == HSV_RGB)
            color_convert_hip_batch_fp32_u8(static_cast<Rpp32f*>(srcPtr),
                                            static_cast<Rpp8u*>(dstPtr),
                                            RPPI_CHN_PLANAR,
                                            3,
                                            rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
 rppi_color_convert_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        RppiColorConvertMode convert_mode,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        color_convert_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               convert_mode,
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        if(convert_mode == RGB_HSV)
            color_convert_hip_batch_u8_fp32(static_cast<Rpp8u*>(srcPtr),
                                            static_cast<Rpp32f*>(dstPtr),
                                            RPPI_CHN_PACKED,
                                            3,
                                            rpp::deref(rppHandle));
        else if(convert_mode == HSV_RGB)
            color_convert_hip_batch_fp32_u8(static_cast<Rpp32f*>(srcPtr),
                                            static_cast<Rpp8u*>(dstPtr),
                                            RPPI_CHN_PACKED,
                                            3,
                                            rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** color_temperature ********************/

RppStatus
rppi_color_temperature_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32s *adjustmentValue,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_int(adjustmentValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        color_temperature_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#elif defined(HIP_COMPILE)
    {
        color_temperature_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32s *adjustmentValue,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_int(adjustmentValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        color_temperature_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        color_temperature_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32s *adjustmentValue,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_int(adjustmentValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        color_temperature_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        color_temperature_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PACKED,
                                    3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** vignette ********************/

RppStatus
rppi_vignette_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *stdDev,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        vignette_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        vignette_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *stdDev,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        vignette_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        vignette_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *stdDev,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        vignette_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        vignette_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** channel_extract ********************/

RppStatus
rppi_channel_extract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u *extractChannelNumber,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(extractChannelNumber, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        channel_extract_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 1);
    }
#elif defined(HIP_COMPILE)
    {
        channel_extract_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_extract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u *extractChannelNumber,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint(extractChannelNumber, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        channel_extract_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        channel_extract_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_extract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u *extractChannelNumber,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint(extractChannelNumber, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        channel_extract_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PACKED,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        channel_extract_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** channel_combine ********************/

RppStatus
rppi_channel_combine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                         RppPtr_t srcPtr2,
                                         RppPtr_t srcPtr3,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        channel_combine_cl_batch(static_cast<cl_mem>(srcPtr1),
                                 static_cast<cl_mem>(srcPtr2),
                                 static_cast<cl_mem>(srcPtr3),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 1);
    }
#elif defined(HIP_COMPILE)
    {
        channel_combine_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                  static_cast<Rpp8u*>(srcPtr2),
                                  static_cast<Rpp8u*>(srcPtr3),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_combine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                         RppPtr_t srcPtr2,
                                         RppPtr_t srcPtr3,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        channel_combine_cl_batch(static_cast<cl_mem>(srcPtr1),
                                 static_cast<cl_mem>(srcPtr2),
                                 static_cast<cl_mem>(srcPtr3),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        channel_combine_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                  static_cast<Rpp8u*>(srcPtr2),
                                  static_cast<Rpp8u*>(srcPtr3),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_channel_combine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                         RppPtr_t srcPtr2,
                                         RppPtr_t srcPtr3,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        channel_combine_cl_batch(static_cast<cl_mem>(srcPtr1),
                                 static_cast<cl_mem>(srcPtr2),
                                 static_cast<cl_mem>(srcPtr3),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PACKED,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        channel_combine_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                  static_cast<Rpp8u*>(srcPtr2),
                                  static_cast<Rpp8u*>(srcPtr3),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** look_up_table ********************/

RppStatus
rppi_look_up_table_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u* lutPtr,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        look_up_table_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),lutPtr,
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#elif defined(HIP_COMPILE)
    {
        look_up_table_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr), lutPtr,
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_look_up_table_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u* lutPtr,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        look_up_table_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),lutPtr,
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        look_up_table_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr), lutPtr,
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_look_up_table_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u* lutPtr,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        look_up_table_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),lutPtr,
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        look_up_table_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr), lutPtr,
                                rpp::deref(rppHandle),
                                RPPI_CHN_PACKED,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

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
#include "rppi_image_augmentations.h"
#include "host_image_augmentations.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** brightness ********************/

RppStatus
rppi_brightness_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32f *alpha,
                                     Rpp32f *beta,
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

    brightness_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 alpha,
                                 beta,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 1,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32f *alpha,
                                     Rpp32f *beta,
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

    brightness_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 alpha,
                                 beta,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32f *alpha,
                                     Rpp32f *beta,
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

    brightness_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 alpha,
                                 beta,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PACKED,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus
rppi_gamma_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32f *gamma,
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

    gamma_correction_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u*>(dstPtr),
                                       gamma,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       1,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32f *gamma,
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

    gamma_correction_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u*>(dstPtr),
                                       gamma,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32f *gamma,
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

    gamma_correction_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u*>(dstPtr),
                                       gamma,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PACKED,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** blend ********************/

RppStatus
rppi_blend_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *alpha,
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

    blend_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            alpha,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *alpha,
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

    blend_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            alpha,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *alpha,
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

    blend_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            alpha,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** blur ********************/

RppStatus
rppi_blur_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *kernelSize,
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

    blur_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           kernelSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *kernelSize,
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

    blur_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           kernelSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *kernelSize,
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

    blur_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           kernelSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** contrast ********************/

RppStatus
rppi_contrast_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u *newMin,
                                   Rpp32u *newMax,
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

    contrast_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               newMin,
                               newMax,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u *newMin,
                                   Rpp32u *newMax,
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

    contrast_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               newMin,
                               newMax,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u *newMin,
                                   Rpp32u *newMax,
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

    contrast_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               newMin,
                               newMax,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** pixelate ********************/

RppStatus
rppi_pixelate_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
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

    pixelate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
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

    pixelate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
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

    pixelate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** jitter ********************/

RppStatus
rppi_jitter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp32u *kernelSize,
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

    jitter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             kernelSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PLANAR,
                             1,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp32u *kernelSize,
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

    jitter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             kernelSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PLANAR,
                             3,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp32u *kernelSize,
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

    jitter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             kernelSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PACKED,
                             3,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** snow ********************/

RppStatus
rppi_snow_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *snowValue,
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

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *snowValue,
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

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *snowValue,
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

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** noise ********************/

RppStatus
rppi_noise_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *noiseProbability,
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

    noise_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            noiseProbability,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_noise_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *noiseProbability,
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

    noise_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            noiseProbability,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_noise_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32f *noiseProbability,
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

    noise_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            noiseProbability,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** random_shadow ********************/

RppStatus
rppi_random_shadow_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *x1,
                                        Rpp32u *y1,
                                        Rpp32u *x2,
                                        Rpp32u *y2,
                                        Rpp32u *numberOfShadows,
                                        Rpp32u *maxSizeX,
                                        Rpp32u *maxSizeY,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    random_shadow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    numberOfShadows,
                                    maxSizeX,
                                    maxSizeY,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    1,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *x1,
                                        Rpp32u *y1,
                                        Rpp32u *x2,
                                        Rpp32u *y2,
                                        Rpp32u *numberOfShadows,
                                        Rpp32u *maxSizeX,
                                        Rpp32u *maxSizeY,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    random_shadow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    numberOfShadows,
                                    maxSizeX,
                                    maxSizeY,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *x1,
                                        Rpp32u *y1,
                                        Rpp32u *x2,
                                        Rpp32u *y2,
                                        Rpp32u *numberOfShadows,
                                        Rpp32u *maxSizeX,
                                        Rpp32u *maxSizeY,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    random_shadow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    numberOfShadows,
                                    maxSizeX,
                                    maxSizeY,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** fog ********************/

RppStatus
rppi_fog_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *fogValue,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fog_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          fogValue,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PLANAR,
                          1,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *fogValue,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fog_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          fogValue,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PLANAR,
                          3,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *fogValue,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fog_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          fogValue,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PACKED,
                          3,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** rain ********************/

RppStatus
rppi_rain_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *rainPercentage,
                               Rpp32u *rainWidth,
                               Rpp32u *rainHeight,
                               Rpp32f *transperancy,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    rain_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           rainPercentage,
                           rainWidth,
                           rainHeight,
                           transperancy,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *rainPercentage,
                               Rpp32u *rainWidth,
                               Rpp32u *rainHeight,
                               Rpp32f *transperancy,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    rain_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           rainPercentage,
                           rainWidth,
                           rainHeight,
                           transperancy,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *rainPercentage,
                               Rpp32u *rainWidth,
                               Rpp32u *rainHeight,
                               Rpp32f *transperancy,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    rain_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           rainPercentage,
                           rainWidth,
                           rainHeight,
                           transperancy,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** random_crop_letterbox ********************/

RppStatus
rppi_random_crop_letterbox_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                RppiSize *dstSize,
                                                RppiSize maxDstSize,
                                                Rpp32u *xRoiBegin,
                                                Rpp32u *xRoiEnd,
                                                Rpp32u *yRoiBegin,
                                                Rpp32u *yRoiEnd,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    random_crop_letterbox_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp8u*>(dstPtr),
                                            dstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                            xRoiBegin,
                                            xRoiEnd,
                                            yRoiBegin,
                                            yRoiEnd,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                            rpp::deref(rppHandle).GetBatchSize(),
                                            RPPI_CHN_PLANAR,
                                            1,
                                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                RppiSize *dstSize,
                                                RppiSize maxDstSize,
                                                Rpp32u *xRoiBegin,
                                                Rpp32u *xRoiEnd,
                                                Rpp32u *yRoiBegin,
                                                Rpp32u *yRoiEnd,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    random_crop_letterbox_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp8u*>(dstPtr),
                                            dstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                            xRoiBegin,
                                            xRoiEnd,
                                            yRoiBegin,
                                            yRoiEnd,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                            rpp::deref(rppHandle).GetBatchSize(),
                                            RPPI_CHN_PLANAR,
                                            3,
                                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                RppiSize *dstSize,
                                                RppiSize maxDstSize,
                                                Rpp32u *xRoiBegin,
                                                Rpp32u *xRoiEnd,
                                                Rpp32u *yRoiBegin,
                                                Rpp32u *yRoiEnd,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    random_crop_letterbox_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp8u*>(dstPtr),
                                            dstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                            xRoiBegin,
                                            xRoiEnd,
                                            yRoiBegin,
                                            yRoiEnd,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                            rpp::deref(rppHandle).GetBatchSize(),
                                            RPPI_CHN_PACKED,
                                            3,
                                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** exposure ********************/

RppStatus
rppi_exposure_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *exposureValue,
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

    exposure_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               exposureValue,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *exposureValue,
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

    exposure_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               exposureValue,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32f *exposureValue,
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

    exposure_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               exposureValue,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** histogram_balance ********************/

RppStatus
rppi_histogram_balance_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_balance_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_histogram_balance_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_balance_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_histogram_balance_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_balance_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8u*>(dstPtr),
                                        rpp::deref(rppHandle).GetBatchSize(),
                                        RPPI_CHN_PACKED,
                                        3,
                                        rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** brightness ********************/

RppStatus
rppi_brightness_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32f *alpha,
                                    Rpp32f *beta,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        brightness_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            1);
    }
#elif defined(HIP_COMPILE)
    {
        brightness_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32f *alpha,
                                    Rpp32f *beta,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        brightness_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        brightness_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32f *alpha,
                                    Rpp32f *beta,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        brightness_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        brightness_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus
rppi_gamma_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *gamma,
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
    copy_param_float(gamma, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gamma_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#elif defined(HIP_COMPILE)
    {
        gamma_correction_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *gamma,
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
    copy_param_float(gamma, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gamma_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        gamma_correction_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *gamma,
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
    copy_param_float(gamma, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gamma_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        gamma_correction_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** blend ********************/

RppStatus
rppi_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *alpha,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        blend_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        blend_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *alpha,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        blend_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        blend_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *alpha,
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
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        blend_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        blend_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** blur ********************/

RppStatus
rppi_blur_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            1);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** contrast ********************/

RppStatus
rppi_contrast_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u *newMin,
                                  Rpp32u *newMax,
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
    copy_param_uint(newMin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(newMax, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        contrast_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        contrast_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u *newMin,
                                  Rpp32u *newMax,
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
    copy_param_uint(newMin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(newMax, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        contrast_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        contrast_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u *newMin,
                                  Rpp32u *newMax,
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
    copy_param_uint(newMin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(newMax, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        contrast_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        contrast_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** pixelate ********************/

RppStatus
rppi_pixelate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
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
        pixelate_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        pixelate_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
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
        pixelate_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        pixelate_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
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
        pixelate_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        pixelate_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** jitter ********************/

RppStatus
rppi_jitter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        jitter_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#elif defined(HIP_COMPILE)
    {
        jitter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        jitter_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#elif defined(HIP_COMPILE)
    {
        jitter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        jitter_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#elif defined(HIP_COMPILE)
    {
        jitter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PACKED,
                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** snow ********************/

RppStatus
rppi_snow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *snowValue,
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
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        snow_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#elif defined(HIP_COMPILE)
    {
        snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *snowValue,
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
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        snow_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *snowValue,
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
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        snow_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** noise ********************/

RppStatus
rppi_noise_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *noiseProbability,
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
    copy_param_float(noiseProbability, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        noise_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        noise_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_noise_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *noiseProbability,
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
    copy_param_float(noiseProbability, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        noise_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        noise_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_noise_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32f *noiseProbability,
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
    copy_param_float(noiseProbability, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        noise_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        noise_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** random_shadow ********************/

RppStatus
rppi_random_shadow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *x1,
                                       Rpp32u *y1,
                                       Rpp32u *x2,
                                       Rpp32u *y2,
                                       Rpp32u *numberOfShadows,
                                       Rpp32u *maxSizeX,
                                       Rpp32u *maxSizeY,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(x1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(numberOfShadows, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeX, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeY, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_shadow_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#elif defined(HIP_COMPILE)
    {
        random_shadow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *x1,
                                       Rpp32u *y1,
                                       Rpp32u *x2,
                                       Rpp32u *y2,
                                       Rpp32u *numberOfShadows,
                                       Rpp32u *maxSizeX,
                                       Rpp32u *maxSizeY,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint(x1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(numberOfShadows, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeX, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeY, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_shadow_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        random_shadow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *x1,
                                       Rpp32u *y1,
                                       Rpp32u *x2,
                                       Rpp32u *y2,
                                       Rpp32u *numberOfShadows,
                                       Rpp32u *maxSizeX,
                                       Rpp32u *maxSizeY,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint(x1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y1, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y2, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(numberOfShadows, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeX, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maxSizeY, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_shadow_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        random_shadow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PACKED,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** fog ********************/

RppStatus
rppi_fog_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32f *fogValue,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(fogValue, rpp::deref(rppHandle), paramIndex++);


#ifdef OCL_COMPILE
    {
        fog_cl_batch(static_cast<cl_mem>(srcPtr),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     1);
    }
#elif defined(HIP_COMPILE)
    {
        fog_hip_batch(static_cast<Rpp8u*>(srcPtr),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32f *fogValue,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(fogValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        fog_cl_batch(static_cast<cl_mem>(srcPtr),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        fog_hip_batch(static_cast<Rpp8u*>(srcPtr),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32f *fogValue,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(fogValue, rpp::deref(rppHandle), paramIndex++);


#ifdef OCL_COMPILE
    {
        fog_cl_batch(static_cast<cl_mem>(srcPtr),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PACKED,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        fog_hip_batch(static_cast<Rpp8u*>(srcPtr),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** rain ********************/

RppStatus
rppi_rain_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *rainPercentage,
                              Rpp32u *rainWidth,
                              Rpp32u *rainHeight,
                              Rpp32f *transperancy,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(rainPercentage, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainWidth, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainHeight, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(transperancy, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        rain_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#elif defined(HIP_COMPILE)
    {
        rain_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *rainPercentage,
                              Rpp32u *rainWidth,
                              Rpp32u *rainHeight,
                              Rpp32f *transperancy,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(rainPercentage, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainWidth, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainHeight, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(transperancy, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        rain_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        rain_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32f *rainPercentage,
                              Rpp32u *rainWidth,
                              Rpp32u *rainHeight,
                              Rpp32f *transperancy,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(rainPercentage, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainWidth, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(rainHeight, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(transperancy, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        rain_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        rain_hip_batch(static_cast<Rpp8u*>(srcPtr),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** random_crop_letterbox ********************/

RppStatus
rppi_random_crop_letterbox_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               RppiSize *dstSize,
                                               RppiSize maxDstSize,
                                               Rpp32u *xRoiBegin,
                                               Rpp32u *xRoiEnd,
                                               Rpp32u *yRoiBegin,
                                               Rpp32u *yRoiEnd,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_crop_letterbox_cl_batch(static_cast<cl_mem>(srcPtr),
                                       static_cast<cl_mem>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PLANAR,
                                       1);
    }
#elif defined(HIP_COMPILE)
    {
        random_crop_letterbox_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                        static_cast<Rpp8u*>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               RppiSize *dstSize,
                                               RppiSize maxDstSize,
                                               Rpp32u *xRoiBegin,
                                               Rpp32u *xRoiEnd,
                                               Rpp32u *yRoiBegin,
                                               Rpp32u *yRoiEnd,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_crop_letterbox_cl_batch(static_cast<cl_mem>(srcPtr),
                                       static_cast<cl_mem>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PLANAR,
                                       3);
    }
#elif defined(HIP_COMPILE)
    {
        random_crop_letterbox_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                        static_cast<Rpp8u*>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               RppiSize *dstSize,
                                               RppiSize maxDstSize,
                                               Rpp32u *xRoiBegin,
                                               Rpp32u *xRoiEnd,
                                               Rpp32u *yRoiBegin,
                                               Rpp32u *yRoiEnd,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        random_crop_letterbox_cl_batch(static_cast<cl_mem>(srcPtr),
                                       static_cast<cl_mem>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PACKED,
                                       3);
    }
#elif defined(HIP_COMPILE)
    {
        random_crop_letterbox_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                        static_cast<Rpp8u*>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PACKED,
                                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** exposure ********************/

RppStatus
rppi_exposure_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *exposureValue,
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
    copy_param_float(exposureValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        exposure_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        exposure_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *exposureValue,
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
    copy_param_float(exposureValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        exposure_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        exposure_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *exposureValue,
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
    copy_param_float(exposureValue, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        exposure_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        exposure_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** histogram_balance ********************/

RppStatus
rppi_histogram_balance_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        histogram_balance_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#elif defined(HIP_COMPILE)
    {
        histogram_balance_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_balance_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        histogram_balance_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        histogram_balance_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_balance_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        histogram_balance_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        histogram_balance_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                    static_cast<Rpp8u*>(dstPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PACKED,
                                    3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

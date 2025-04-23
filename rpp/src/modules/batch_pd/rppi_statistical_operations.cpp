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
#include "rppi_statistical_operations.h"
#include "host_statistical_operations.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** thresholding ********************/

RppStatus
rppi_thresholding_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u *minThreshold,
                                       Rpp8u *maxThreshold,
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

    thresholding_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   minThreshold,
                                   maxThreshold,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PLANAR,
                                   1,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_thresholding_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u *minThreshold,
                                       Rpp8u *maxThreshold,
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

    thresholding_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   minThreshold,
                                   maxThreshold,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PLANAR,
                                   3,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_thresholding_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp8u *minThreshold,
                                       Rpp8u *maxThreshold,
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

    thresholding_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   minThreshold,
                                   maxThreshold,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PACKED,
                                   3,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** min ********************/

RppStatus
rppi_min_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    min_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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
rppi_min_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    min_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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
rppi_min_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    min_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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

/******************** max ********************/

RppStatus
rppi_max_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    max_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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
rppi_max_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    max_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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
rppi_max_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
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

    max_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
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

/******************** min_max_loc ********************/

RppStatus
rppi_min_max_loc_u8_pln1_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp8u* min,
                              Rpp8u* max,
                              Rpp32u* minLoc,
                              Rpp32u* maxLoc,
                              rppHandle_t rppHandle)
{
    min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            min,
                            max,
                            minLoc,
                            maxLoc,
                            RPPI_CHN_PLANAR,
                            1);

    return RPP_SUCCESS;
}

RppStatus
rppi_min_max_loc_u8_pln3_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp8u* min,
                              Rpp8u* max,
                              Rpp32u* minLoc,
                              Rpp32u* maxLoc,
                              rppHandle_t rppHandle)
{
    min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            min,
                            max,
                            minLoc,
                            maxLoc,
                            RPPI_CHN_PLANAR,
                            3);

    return RPP_SUCCESS;
}

RppStatus
rppi_min_max_loc_u8_pkd3_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp8u* min,
                              Rpp8u* max,
                              Rpp32u* minLoc,
                              Rpp32u* maxLoc,
                              rppHandle_t rppHandle)
{
    min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            min,
                            max,
                            minLoc,
                            maxLoc,
                            RPPI_CHN_PACKED,
                            3);

    return RPP_SUCCESS;
}

/******************** integral ********************/

RppStatus
rppi_integral_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    integral_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp32u*>(dstPtr),
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_integral_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    integral_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp32u*>(dstPtr),
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_integral_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    integral_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp32u*>(dstPtr),
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** histogram_equalization ********************/

RppStatus
rppi_histogram_equalization_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_equalization_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_histogram_equalization_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_equalization_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_histogram_equalization_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    histogram_equalization_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PACKED,
                                             3,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** mean_stddev ********************/

RppStatus
rppi_mean_stddev_u8_pln1_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp32f *mean,
                              Rpp32f *stddev,
                              rppHandle_t rppHandle)
 {
    mean_stddev_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            mean,
                            stddev,
                            RPPI_CHN_PLANAR,
                            1);

    return RPP_SUCCESS;
}

RppStatus
rppi_mean_stddev_u8_pln3_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp32f *mean,
                              Rpp32f *stddev,
                              rppHandle_t rppHandle)
 {
    mean_stddev_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            mean,
                            stddev,
                            RPPI_CHN_PLANAR,
                            3);

    return RPP_SUCCESS;
}

RppStatus
rppi_mean_stddev_u8_pkd3_host(RppPtr_t srcPtr,
                              RppiSize srcSize,
                              Rpp32f *mean,
                              Rpp32f *stddev,
                              rppHandle_t rppHandle)
 {
     mean_stddev_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             mean,
                             stddev,
                             RPPI_CHN_PACKED,
                             3);

    return RPP_SUCCESS;
}

/******************** histogram ********************/

RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr,
                            RppiSize srcSize,
                            Rpp32u* outputHistogram,
                            Rpp32u bins,
                            rppHandle_t rppHandle)
{
     histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           outputHistogram,
                           bins,
                           1);

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr,
                            RppiSize srcSize,
                            Rpp32u* outputHistogram,
                            Rpp32u bins,
                            rppHandle_t rppHandle)
{
     histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           outputHistogram,
                           bins,
                           3);

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr,
                            RppiSize srcSize,
                            Rpp32u* outputHistogram,
                            Rpp32u bins,
                            rppHandle_t rppHandle)
{
     histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           outputHistogram,
                           bins,
                           3);

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** thresholding ********************/

RppStatus
rppi_thresholding_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp8u *minThreshold,
                                      Rpp8u *maxThreshold,
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
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        thresholding_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              1);
    }
#elif defined(HIP_COMPILE)
    {
        thresholding_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_thresholding_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp8u *minThreshold,
                                      Rpp8u *maxThreshold,
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
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        thresholding_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        thresholding_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_thresholding_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp8u *minThreshold,
                                      Rpp8u *maxThreshold,
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
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        thresholding_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        thresholding_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** min ********************/

RppStatus
rppi_min_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        min_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     1);
    }
#elif defined(HIP_COMPILE)
    {
        min_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_min_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        min_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        min_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_min_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        min_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PACKED,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        min_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                      static_cast<Rpp8u*>(srcPtr2),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** max ********************/

RppStatus
rppi_max_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        max_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     1);
    }
#elif defined(HIP_COMPILE)
    {
        max_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_max_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        max_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        max_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_max_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
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
        max_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PACKED,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        max_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                      static_cast<Rpp8u*>(srcPtr2),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** min_max_loc ********************/

RppStatus
rppi_min_max_loc_u8_pln1_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp8u* min,
                             Rpp8u* max,
                             Rpp32u* minLoc,
                             Rpp32u* maxLoc,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        min_max_loc_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       min,
                       max,
                       minLoc,
                       maxLoc,
                       RPPI_CHN_PLANAR,
                       1,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        min_max_loc_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        min,
                        max,
                        minLoc,
                        maxLoc,
                        RPPI_CHN_PLANAR,
                        1,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}


RppStatus
rppi_min_max_loc_u8_pln3_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp8u* min,
                             Rpp8u* max,
                             Rpp32u* minLoc,
                             Rpp32u* maxLoc,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        min_max_loc_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       min,
                       max,
                       minLoc,
                       maxLoc,
                       RPPI_CHN_PLANAR,
                       3,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        min_max_loc_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        min,
                        max,
                        minLoc,
                        maxLoc,
                        RPPI_CHN_PLANAR,
                        3,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_min_max_loc_u8_pkd3_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp8u* min,
                             Rpp8u* max,
                             Rpp32u* minLoc,
                             Rpp32u* maxLoc,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        min_max_loc_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       min,
                       max,
                       minLoc,
                       maxLoc,
                       RPPI_CHN_PACKED,
                       3,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        min_max_loc_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        min,
                        max,
                        minLoc,
                        maxLoc,
                        RPPI_CHN_PACKED,
                        3,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

/******************** integral ********************/

RppStatus
rppi_integral_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        integral_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        integral_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp32u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_integral_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        integral_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        integral_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp32u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_integral_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        integral_cl_batch(static_cast<cl_mem>(srcPtr),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        integral_hip_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp32u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** histogram_equalization ********************/

RppStatus
rppi_histogram_equalization_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
rppi_histogram_equalization_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
rppi_histogram_equalization_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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

/******************** mean_stddev ********************/

RppStatus
rppi_mean_stddev_u8_pln1_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp32f *mean,
                             Rpp32f *stddev,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        mean_stddev_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       mean,
                       stddev,
                       RPPI_CHN_PLANAR,
                       1,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        mean_stddev_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        mean,
                        stddev,
                        RPPI_CHN_PLANAR,
                        1,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_mean_stddev_u8_pln3_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp32f *mean,
                             Rpp32f *stddev,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        mean_stddev_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       mean,
                       stddev,
                       RPPI_CHN_PLANAR,
                       3,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        mean_stddev_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        mean,
                        stddev,
                        RPPI_CHN_PLANAR,
                        3,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_mean_stddev_u8_pkd3_gpu(RppPtr_t srcPtr,
                             RppiSize srcSize,
                             Rpp32f *mean,
                             Rpp32f *stddev,
                             rppHandle_t rppHandle)
{
    #ifdef OCL_COMPILE
    {
        mean_stddev_cl(static_cast<cl_mem>(srcPtr),
                       srcSize,
                       mean,
                       stddev,
                       RPPI_CHN_PACKED,
                       3,
                       rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {
        mean_stddev_hip(static_cast<Rpp8u*>(srcPtr),
                        srcSize,
                        mean,
                        stddev,
                        RPPI_CHN_PACKED,
                        3,
                        rpp::deref(rppHandle));
    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

/******************** histogram ********************/

RppStatus
rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr,
                           RppiSize srcSize,
                           Rpp32u *outputHistogram,
                           Rpp32u bins,
                           rppHandle_t rppHandle)
{
     #ifdef OCL_COMPILE
    {
        histogram_cl(static_cast<cl_mem>(srcPtr),
                     srcSize,
                     outputHistogram,
                     bins,
                     RPPI_CHN_PLANAR,
                     1,
                     rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {

    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pln3_gpu(RppPtr_t srcPtr,
                           RppiSize srcSize,
                           Rpp32u *outputHistogram,
                           Rpp32u bins,
                           rppHandle_t rppHandle)
{
     #ifdef OCL_COMPILE
    {
        histogram_cl(static_cast<cl_mem>(srcPtr),
                     srcSize,
                     outputHistogram,
                     bins,
                     RPPI_CHN_PLANAR,
                     3,
                     rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {

    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pkd3_gpu(RppPtr_t srcPtr,
                           RppiSize srcSize,
                           Rpp32u *outputHistogram,
                           Rpp32u bins,
                           rppHandle_t rppHandle)
{
     #ifdef OCL_COMPILE
    {
        histogram_cl(static_cast<cl_mem>(srcPtr),
                     srcSize,
                     outputHistogram,
                     bins,
                     RPPI_CHN_PACKED,
                     3,
                     rpp::deref(rppHandle));
    }
    #elif defined(HIP_COMPILE)
    {

    }
    #endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

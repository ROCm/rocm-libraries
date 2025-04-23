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
#include "rppi_logical_operations.h"
#include "host_logical_operations.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** bitwise_AND ********************/

RppStatus
rppi_bitwise_AND_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
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

    bitwise_AND_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_bitwise_AND_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
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

    bitwise_AND_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_bitwise_AND_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
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

    bitwise_AND_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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

/******************** bitwise_NOT ********************/

RppStatus
rppi_bitwise_NOT_u8_pln1_batchPD_host(RppPtr_t srcPtr,
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

    bitwise_NOT_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_bitwise_NOT_u8_pln3_batchPD_host(RppPtr_t srcPtr,
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

    bitwise_NOT_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_bitwise_NOT_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
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

    bitwise_NOT_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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

/******************** exclusive_OR ********************/

RppStatus
rppi_exclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
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

    exclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_exclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
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

    exclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_exclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
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

    exclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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

/******************** inclusive_OR ********************/

RppStatus
rppi_inclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
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

    inclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_inclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
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

    inclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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
rppi_inclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
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

    inclusive_OR_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
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

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** bitwise_AND ********************/

RppStatus
rppi_bitwise_AND_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
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
        bitwise_AND_cl_batch(static_cast<cl_mem>(srcPtr1),
                             static_cast<cl_mem>(srcPtr2),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_AND_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_bitwise_AND_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
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
        bitwise_AND_cl_batch(static_cast<cl_mem>(srcPtr1),
                             static_cast<cl_mem>(srcPtr2),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_AND_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_bitwise_AND_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
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
        bitwise_AND_cl_batch(static_cast<cl_mem>(srcPtr1),
                             static_cast<cl_mem>(srcPtr2),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_AND_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                              static_cast<Rpp8u*>(srcPtr2),
                              static_cast<Rpp8u*>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** bitwise_NOT ********************/

RppStatus
rppi_bitwise_NOT_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        bitwise_NOT_cl_batch(static_cast<cl_mem>(srcPtr),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_NOT_hip_batch(static_cast<Rpp8u*>(srcPtr),
                              static_cast<Rpp8u*>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        bitwise_NOT_cl_batch(static_cast<cl_mem>(srcPtr),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_NOT_hip_batch(static_cast<Rpp8u*>(srcPtr),
                              static_cast<Rpp8u*>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        bitwise_NOT_cl_batch(static_cast<cl_mem>(srcPtr),
                             static_cast<cl_mem>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#elif defined(HIP_COMPILE)
    {
        bitwise_NOT_hip_batch(static_cast<Rpp8u*>(srcPtr),
                              static_cast<Rpp8u*>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** exclusive_OR ********************/

RppStatus
rppi_exclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
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
        exclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              1);
    }
#elif defined(HIP_COMPILE)
    {
        exclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_exclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
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
        exclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        exclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_exclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
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
        exclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        exclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** inclusive_OR ********************/

RppStatus
rppi_inclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
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
        inclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              1);
    }
#elif defined(HIP_COMPILE)
    {
        inclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_inclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
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
        inclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        inclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
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
rppi_inclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
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
        inclusive_OR_cl_batch(static_cast<cl_mem>(srcPtr1),
                              static_cast<cl_mem>(srcPtr2),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        inclusive_OR_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

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
#include "rppi_computer_vision.h"
#include "host_computer_vision.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** local_binary_pattern ********************/

RppStatus
rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr,
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

    local_binary_pattern_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr,
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

    local_binary_pattern_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
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

    local_binary_pattern_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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

/******************** data_object_copy ********************/

RppStatus
rppi_data_object_copy_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    data_object_copy_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_data_object_copy_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    data_object_copy_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
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
rppi_data_object_copy_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    data_object_copy_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u*>(dstPtr),
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PACKED,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** gaussian_image_pyramid ********************/

RppStatus
rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    gaussian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             stdDev,
                                             kernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PLANAR,
                                             1,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    gaussian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             stdDev,
                                             kernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PLANAR,
                                             3,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    gaussian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             stdDev,
                                             kernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PACKED,
                                             3,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** laplacian_image_pyramid ********************/

RppStatus
rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                                  RppiSize *srcSize,
                                                  RppiSize maxSrcSize,
                                                  RppPtr_t dstPtr,
                                                  Rpp32f *stdDev,
                                                  Rpp32u *kernelSize,
                                                  Rpp32u nbatchSize,
                                                  rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp8u*>(dstPtr),
                                              stdDev,
                                              kernelSize,
                                              rpp::deref(rppHandle).GetBatchSize(),
                                              RPPI_CHN_PLANAR,
                                              1,
                                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                                  RppiSize *srcSize,
                                                  RppiSize maxSrcSize,
                                                  RppPtr_t dstPtr,
                                                  Rpp32f *stdDev,
                                                  Rpp32u *kernelSize,
                                                  Rpp32u nbatchSize,
                                                  rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp8u*>(dstPtr),
                                              stdDev,
                                              kernelSize,
                                              rpp::deref(rppHandle).GetBatchSize(),
                                              RPPI_CHN_PLANAR,
                                              3,
                                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                  RppiSize *srcSize,
                                                  RppiSize maxSrcSize,
                                                  RppPtr_t dstPtr,
                                                  Rpp32f *stdDev,
                                                  Rpp32u *kernelSize,
                                                  Rpp32u nbatchSize,
                                                  rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp8u*>(dstPtr),
                                              stdDev,
                                              kernelSize,
                                              rpp::deref(rppHandle).GetBatchSize(),
                                              RPPI_CHN_PACKED,
                                              3,
                                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** canny_edge_detector ********************/

RppStatus
rppi_canny_edge_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp8u *minThreshold,
                                              Rpp8u *maxThreshold,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    canny_edge_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          minThreshold,
                                          maxThreshold,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          1,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_canny_edge_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp8u *minThreshold,
                                              Rpp8u *maxThreshold,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    canny_edge_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          minThreshold,
                                          maxThreshold,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_canny_edge_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp8u *minThreshold,
                                              Rpp8u *maxThreshold,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    canny_edge_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          minThreshold,
                                          maxThreshold,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PACKED,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** harris_corner_detector ********************/

RppStatus
rppi_harris_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u *gaussianKernelSize,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32f *kValue,
                                                 Rpp32f *threshold,
                                                 Rpp32u *nonmaxKernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    harris_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             gaussianKernelSize,
                                             stdDev,
                                             kernelSize,
                                             kValue,
                                             threshold,
                                             nonmaxKernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PLANAR,
                                             1,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_harris_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u *gaussianKernelSize,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32f *kValue,
                                                 Rpp32f *threshold,
                                                 Rpp32u *nonmaxKernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    harris_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             gaussianKernelSize,
                                             stdDev,
                                             kernelSize,
                                             kValue,
                                             threshold,
                                             nonmaxKernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PLANAR,
                                             3,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_harris_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32u *gaussianKernelSize,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32f *kValue,
                                                 Rpp32f *threshold,
                                                 Rpp32u *nonmaxKernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    harris_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u*>(dstPtr),
                                             gaussianKernelSize,
                                             stdDev,
                                             kernelSize,
                                             kValue,
                                             threshold,
                                             nonmaxKernelSize,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             RPPI_CHN_PACKED,
                                             3,
                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** tensor_convert_bit_depth ********************/

RppStatus
rppi_tensor_convert_bit_depth_u8s8_host(RppPtr_t srcPtr,
                                        RppPtr_t dstPtr,
                                        Rpp32u tensorDimension,
                                        RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp8s>(static_cast<Rpp8u*>(srcPtr),
                                                static_cast<Rpp8s*>(dstPtr),
                                                1,
                                                tensorDimension,
                                                static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_convert_bit_depth_u8u16_host(RppPtr_t srcPtr,
                                         RppPtr_t dstPtr,
                                         Rpp32u tensorDimension,
                                         RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp16u>(static_cast<Rpp8u*>(srcPtr),
                                                 static_cast<Rpp16u*>(dstPtr),
                                                 2,
                                                 tensorDimension,
                                                 static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_convert_bit_depth_u8s16_host(RppPtr_t srcPtr,
                                         RppPtr_t dstPtr,
                                         Rpp32u tensorDimension,
                                         RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp16s>(static_cast<Rpp8u*>(srcPtr),
                                                 static_cast<Rpp16s*>(dstPtr),
                                                 3,
                                                 tensorDimension,
                                                 static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

/******************** fast_corner_detector ********************/

RppStatus
rppi_fast_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u *numOfPixels,
                                               Rpp8u *threshold,
                                               Rpp32u *nonmaxKernelSize,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fast_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp8u*>(dstPtr),
                                           numOfPixels,
                                           threshold,
                                           nonmaxKernelSize,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           RPPI_CHN_PLANAR,
                                           1,
                                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fast_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u *numOfPixels,
                                               Rpp8u *threshold,
                                               Rpp32u *nonmaxKernelSize,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fast_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp8u*>(dstPtr),
                                           numOfPixels,
                                           threshold,
                                           nonmaxKernelSize,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           RPPI_CHN_PLANAR,
                                           3,
                                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fast_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u *numOfPixels,
                                               Rpp8u *threshold,
                                               Rpp32u *nonmaxKernelSize,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fast_corner_detector_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp8u*>(dstPtr),
                                           numOfPixels,
                                           threshold,
                                           nonmaxKernelSize,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           RPPI_CHN_PACKED,
                                           3,
                                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** reconstruction_laplacian_image_pyramid ********************/

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                                                 RppiSize *srcSize1,
                                                                 RppiSize maxSrcSize1,
                                                                 RppPtr_t srcPtr2,
                                                                 RppiSize *srcSize2,
                                                                 RppiSize maxSrcSize2,
                                                                 RppPtr_t dstPtr,
                                                                 Rpp32f *stdDev,
                                                                 Rpp32u *kernelSize,
                                                                 Rpp32u nbatchSize,
                                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                                             srcSize1,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                             static_cast<Rpp8u*>(srcPtr2),
                                                             srcSize2,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                             static_cast<Rpp8u*>(dstPtr),
                                                             stdDev,
                                                             kernelSize,
                                                             rpp::deref(rppHandle).GetBatchSize(),
                                                             RPPI_CHN_PLANAR,
                                                             1,
                                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                                                 RppiSize *srcSize1,
                                                                 RppiSize maxSrcSize1,
                                                                 RppPtr_t srcPtr2,
                                                                 RppiSize *srcSize2,
                                                                 RppiSize maxSrcSize2,
                                                                 RppPtr_t dstPtr,
                                                                 Rpp32f *stdDev,
                                                                 Rpp32u *kernelSize,
                                                                 Rpp32u nbatchSize,
                                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                                             srcSize1,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                             static_cast<Rpp8u*>(srcPtr2),
                                                             srcSize2,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                             static_cast<Rpp8u*>(dstPtr),
                                                             stdDev,
                                                             kernelSize,
                                                             rpp::deref(rppHandle).GetBatchSize(),
                                                             RPPI_CHN_PLANAR,
                                                             3,
                                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                                                 RppiSize *srcSize1,
                                                                 RppiSize maxSrcSize1,
                                                                 RppPtr_t srcPtr2,
                                                                 RppiSize *srcSize2,
                                                                 RppiSize maxSrcSize2,
                                                                 RppPtr_t dstPtr,
                                                                 Rpp32f *stdDev,
                                                                 Rpp32u *kernelSize,
                                                                 Rpp32u nbatchSize,
                                                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                                             srcSize1,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                             static_cast<Rpp8u*>(srcPtr2),
                                                             srcSize2,
                                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                             static_cast<Rpp8u*>(dstPtr),
                                                             stdDev,
                                                             kernelSize,
                                                             rpp::deref(rppHandle).GetBatchSize(),
                                                             RPPI_CHN_PACKED,
                                                             3,
                                                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** control_flow ********************/

RppStatus
rpp_bool_control_flow(bool num1,
                      bool num2,
                      bool *output,
                      RppOp operation,
                      rppHandle_t rppHandle)
{
    if(operation == RPP_SCALAR_OP_AND)
        *output = num1 & num2;
    if(operation == RPP_SCALAR_OP_OR)
        *output = num1 | num2;
    if(operation == RPP_SCALAR_OP_XOR)
        *output = num1 ^ num2;
    if(operation == RPP_SCALAR_OP_NAND)
        *output = !(num1 & num2);

    return RPP_SUCCESS;
}

RppStatus
rpp_u8_control_flow(Rpp8u num1,
                    Rpp8u num2,
                    Rpp8u *output,
                    RppOp operation,
                    rppHandle_t rppHandle)
{
    if(operation == RPP_SCALAR_OP_ADD)
        *output = num1 + num2;
    if(operation == RPP_SCALAR_OP_SUBTRACT)
        *output = num1 - num2;
    if(operation == RPP_SCALAR_OP_MULTIPLY)
        *output = num1 * num2;
    if(operation == RPP_SCALAR_OP_MODULUS)
        *output = num1 % num2;
    if(operation == RPP_SCALAR_OP_DIVIDE)
        *output = num1 / num2;
    if(operation == RPP_SCALAR_OP_MIN)
        *output = std::min(num1, num2);
    if(operation == RPP_SCALAR_OP_MAX)
        *output = std::max(num1, num2);
    if(operation == RPP_SCALAR_OP_EQUAL)
        *output =(num1 == num2);
    if(operation == RPP_SCALAR_OP_NOTEQUAL)
        *output =(num1 != num2);
    if(operation == RPP_SCALAR_OP_LESS)
        *output =(num1 < num2);
    if(operation == RPP_SCALAR_OP_LESSEQ)
        *output =(num1 <= num2);
    if(operation == RPP_SCALAR_OP_GREATER)
        *output =(num1 > num2);
    if(operation == RPP_SCALAR_OP_GREATEREQ)
        *output =(num1 >= num2);

    return RPP_SUCCESS;
}

/******************** hough_lines ********************/

RppStatus
rppi_hough_lines_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t lines,
                                      Rpp32f *rho,
                                      Rpp32f *theta,
                                      Rpp32u *threshold,
                                      Rpp32u *minLineLength,
                                      Rpp32u *maxLineGap,
                                      Rpp32u *linesMax,
                                      Rpp32u nbatchSize,
                                      rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hough_lines_host_batch<Rpp8u, Rpp32u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp32u*>(lines),
                                          rho,
                                          theta,
                                          threshold,
                                          minLineLength,
                                          maxLineGap,
                                          linesMax,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          1,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** hog ********************/

RppStatus
rppi_hog_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t binsTensor,
                              Rpp32u *binsTensorLength,
                              RppiSize *kernelSize,
                              RppiSize *windowSize,
                              Rpp32u *windowStride,
                              Rpp32u *numOfBins,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hog_host_batch<Rpp8u, Rpp32u>(static_cast<Rpp8u*>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp32u*>(binsTensor),
                                  binsTensorLength,
                                  kernelSize,
                                  windowSize,
                                  windowStride,
                                  numOfBins,
                                  rpp::deref(rppHandle).GetBatchSize(),
                                  RPPI_CHN_PLANAR,
                                  1,
                                  rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** remap ********************/

RppStatus
rppi_remap_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *rowRemapTable,
                                Rpp32u *colRemapTable,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    remap_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rowRemapTable,
                            colRemapTable,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_remap_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *rowRemapTable,
                                Rpp32u *colRemapTable,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    remap_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rowRemapTable,
                            colRemapTable,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_remap_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *rowRemapTable,
                                Rpp32u *colRemapTable,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    remap_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rowRemapTable,
                            colRemapTable,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** tensor_matrix_multiply ********************/

RppStatus
rppi_tensor_matrix_multiply_u8_host(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppPtr_t dstPtr,
                                    RppPtr_t tensorDimensionValues1,
                                    RppPtr_t tensorDimensionValues2,
                                    rppHandle_t rppHandle)
{
    tensor_matrix_multiply_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                       static_cast<Rpp8u*>(srcPtr2),
                                       static_cast<Rpp8u*>(dstPtr),
                                       static_cast<Rpp32u*>(tensorDimensionValues1),
                                       static_cast<Rpp32u*>(tensorDimensionValues2));

    return RPP_SUCCESS;
}

/******************** tensor_transpose ********************/

RppStatus
rppi_tensor_transpose_u8_host(RppPtr_t srcPtr,
                              RppPtr_t dstPtr,
                              Rpp32u *shape,
                              Rpp32u *perm,
                              rppHandle_t rppHandle)
{
    tensor_transpose_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 static_cast<Rpp8u*>(dstPtr),
                                 shape,
                                 perm);

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f16_host(RppPtr_t srcPtr,
                               RppPtr_t dstPtr,
                               Rpp32u *shape,
                               Rpp32u *perm,
                               rppHandle_t rppHandle)
{
    tensor_transpose_host<Rpp16f>(static_cast<Rpp16f*>(srcPtr),
                                  static_cast<Rpp16f*>(dstPtr),
                                  shape,
                                  perm);

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f32_host(RppPtr_t srcPtr,
                               RppPtr_t dstPtr,
                               Rpp32u *shape,
                               Rpp32u *perm,
                               rppHandle_t rppHandle)
{
    tensor_transpose_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr),
                                  static_cast<Rpp32f*>(dstPtr),
                                  shape,
                                  perm);

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_i8_host(RppPtr_t srcPtr,
                              RppPtr_t dstPtr,
                              Rpp32u *shape,
                              Rpp32u *perm,
                              rppHandle_t rppHandle)
{
    tensor_transpose_host<Rpp8s>(static_cast<Rpp8s*>(srcPtr),
                                 static_cast<Rpp8s*>(dstPtr),
                                 shape,
                                 perm);

    return RPP_SUCCESS;
}

/******************** convert_bit_depth ********************/

RppStatus
rppi_convert_bit_depth_u8s8_pln1_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u*>(srcPtr),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp8s*>(dstPtr),
                                               1,
                                               rpp::deref(rppHandle).GetBatchSize(),
                                               RPPI_CHN_PLANAR,
                                               1,
                                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln1_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16u*>(dstPtr),
                                                2,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                RPPI_CHN_PLANAR,
                                                1,
                                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln1_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16s*>(dstPtr),
                                                3,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                RPPI_CHN_PLANAR,
                                                1,
                                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pln3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u*>(srcPtr),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp8s*>(dstPtr),
                                               1,
                                               rpp::deref(rppHandle).GetBatchSize(),
                                               RPPI_CHN_PLANAR,
                                               3,
                                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16u*>(dstPtr),
                                                2,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                RPPI_CHN_PLANAR,
                                                3,
                                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16s*>(dstPtr),
                                                3,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                RPPI_CHN_PLANAR,
                                                3,
                                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u*>(srcPtr),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp8s*>(dstPtr),
                                               1,
                                               rpp::deref(rppHandle).GetBatchSize(),
                                               RPPI_CHN_PACKED,
                                               3,
                                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pkd3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16u*>(dstPtr),
                                                2,
                                                rpp::deref(rppHandle).GetBatchSize(),
                                                RPPI_CHN_PACKED,
                                                3,
                                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pkd3_batchPD_host(RppPtr_t srcPtr,
                                               RppiSize *srcSize,
                                               RppiSize maxSrcSize,
                                               RppPtr_t dstPtr,
                                               Rpp32u nbatchSize,
                                               rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(static_cast<Rpp8u*>(srcPtr),
                                                srcSize,
                                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                static_cast<Rpp16s*>(dstPtr),
                                                3,
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

/******************** local_binary_pattern ********************/

RppStatus
rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        local_binary_pattern_cl_batch(static_cast<cl_mem>(srcPtr),
                                      static_cast<cl_mem>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      1);
    }
#elif defined(HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                       static_cast<Rpp8u*>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PLANAR,
                                       1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        local_binary_pattern_cl_batch(static_cast<cl_mem>(srcPtr),
                                      static_cast<cl_mem>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      3);
    }
#elif defined(HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                       static_cast<Rpp8u*>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PLANAR,
                                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        local_binary_pattern_cl_batch(static_cast<cl_mem>(srcPtr),
                                      static_cast<cl_mem>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PACKED,
                                      3);
    }
#elif defined(HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                       static_cast<Rpp8u*>(dstPtr),
                                       rpp::deref(rppHandle),
                                       RPPI_CHN_PACKED,
                                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** data_object_copy ********************/

RppStatus
rppi_data_object_copy_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        data_object_copy_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#elif defined(HIP_COMPILE)
    {
        data_object_copy_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        data_object_copy_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        data_object_copy_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        data_object_copy_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        data_object_copy_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** gaussian_image_pyramid ********************/

RppStatus
rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        1);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        3);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PACKED,
                                        3);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PACKED,
                                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** laplacian_image_pyramid ********************/

RppStatus
rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                         static_cast<cl_mem>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         1);
    }
#elif defined(HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle),
                                          RPPI_CHN_PLANAR,
                                          1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                         static_cast<cl_mem>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         3);
    }
#elif defined(HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle),
                                          RPPI_CHN_PLANAR,
                                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *stdDev,
                                                 Rpp32u *kernelSize,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(static_cast<cl_mem>(srcPtr),
                                         static_cast<cl_mem>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PACKED,
                                         3);
    }
#elif defined(HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle),
                                          RPPI_CHN_PACKED,
                                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** canny_edge_detector ********************/

RppStatus
rppi_canny_edge_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp8u *minThreshold,
                                             Rpp8u *maxThreshold,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#elif defined(HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_canny_edge_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp8u *minThreshold,
                                             Rpp8u *maxThreshold,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp8u *minThreshold,
                                             Rpp8u *maxThreshold,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uchar(minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PACKED,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** harris_corner_detector ********************/

RppStatus
rppi_harris_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32u *gaussianKernelSize,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32f *kValue,
                                                Rpp32f *threshold,
                                                Rpp32u *nonmaxKernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        1);
    }
#elif defined(HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_harris_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32u *gaussianKernelSize,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32f *kValue,
                                                Rpp32f *threshold,
                                                Rpp32u *nonmaxKernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint(gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PLANAR,
                                        3);
    }
#elif defined(HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PLANAR,
                                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                                RppiSize *srcSize,
                                                RppiSize maxSrcSize,
                                                RppPtr_t dstPtr,
                                                Rpp32u *gaussianKernelSize,
                                                Rpp32f *stdDev,
                                                Rpp32u *kernelSize,
                                                Rpp32f *kValue,
                                                Rpp32f *threshold,
                                                Rpp32u *nonmaxKernelSize,
                                                Rpp32u nbatchSize,
                                                rppHandle_t rppHandle)
{
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint(gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(static_cast<cl_mem>(srcPtr),
                                        static_cast<cl_mem>(dstPtr),
                                        rpp::deref(rppHandle),
                                        RPPI_CHN_PACKED,
                                        3);
    }
#elif defined(HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                         static_cast<Rpp8u*>(dstPtr),
                                         rpp::deref(rppHandle),
                                         RPPI_CHN_PACKED,
                                         3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** remap ********************/

RppStatus
rppi_remap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *rowRemapTable,
                               Rpp32u *colRemapTable,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rowRemapTable,
                       colRemapTable,
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        remap_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rowRemapTable,
                        colRemapTable,
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_remap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *rowRemapTable,
                               Rpp32u *colRemapTable,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rowRemapTable,
                       colRemapTable,
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        remap_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rowRemapTable,
                        colRemapTable,
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_remap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *rowRemapTable,
                               Rpp32u *colRemapTable,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rowRemapTable,
                       colRemapTable,
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        remap_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rowRemapTable,
                        colRemapTable,
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** tensor_matrix_multiply ********************/

RppStatus
rppi_tensor_matrix_multiply_u8_gpu(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppPtr_t dstPtr,
                                   RppPtr_t tensorDimensionValues1,
                                   RppPtr_t tensorDimensionValues2,
                                   rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        tensor_matrix_multiply_cl(static_cast<cl_mem>(srcPtr1),
                                  static_cast<cl_mem>(srcPtr2),
                                  static_cast<Rpp32u*>(tensorDimensionValues1),
                                  static_cast<Rpp32u*>(tensorDimensionValues2),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_matrix_multiply_hip(static_cast<Rpp8u*>(srcPtr1),
                                   static_cast<Rpp8u*>(srcPtr2),
                                   static_cast<Rpp32u*>(tensorDimensionValues1),
                                   static_cast<Rpp32u*>(tensorDimensionValues2),
                                   static_cast<Rpp8u*>(dstPtr),
                                   rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** tensor_transpose ********************/

RppStatus
rppi_tensor_transpose_u8_gpu(RppPtr_t srcPtr,
                             RppPtr_t dstPtr,
                             RppPtr_t shape,
                             RppPtr_t perm,
                             rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            static_cast<Rpp32u*>(shape),
                            static_cast<Rpp32u*>(perm),
                            RPPTensorDataType::U8,
                            rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_transpose_hip_u8(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                static_cast<Rpp32u*>(shape),
                                static_cast<Rpp32u*>(perm),
                                rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f16_gpu(RppPtr_t srcPtr,
                              RppPtr_t dstPtr,
                              RppPtr_t shape,
                              RppPtr_t perm,
                              rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            static_cast<Rpp32u*>(shape),
                            static_cast<Rpp32u*>(perm),
                            RPPTensorDataType::FP16,
                            rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_transpose_hip_fp16(static_cast<Rpp16f*>(srcPtr),
                                  static_cast<Rpp16f*>(dstPtr),
                                  static_cast<Rpp32u*>(shape),
                                  static_cast<Rpp32u*>(perm),
                                  rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f32_gpu(RppPtr_t srcPtr,
                              RppPtr_t dstPtr,
                              RppPtr_t shape,
                              RppPtr_t perm,
                              rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            static_cast<Rpp32u*>(shape),
                            static_cast<Rpp32u*>(perm),
                            RPPTensorDataType::FP32,
                            rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_transpose_hip_fp32(static_cast<Rpp32f*>(srcPtr),
                                  static_cast<Rpp32f*>(dstPtr),
                                  static_cast<Rpp32u*>(shape),
                                  static_cast<Rpp32u*>(perm),
                                  rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_i8_gpu(RppPtr_t srcPtr,
                             RppPtr_t dstPtr,
                             RppPtr_t shape,
                             RppPtr_t perm,
                             rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            static_cast<Rpp32u*>(shape),
                            static_cast<Rpp32u*>(perm),
                            RPPTensorDataType::I8,
                            rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_transpose_hip_i8(static_cast<Rpp8s*>(srcPtr),
                                static_cast<Rpp8s*>(dstPtr),
                                static_cast<Rpp32u*>(shape),
                                static_cast<Rpp32u*>(perm),
                                rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** convert_bit_depth ********************/

RppStatus
rppi_convert_bit_depth_u8s8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   1,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   2,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   3,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   1,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   2,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   3,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   1,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   2,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        convert_bit_depth_cl_batch(static_cast<cl_mem>(srcPtr),
                                   static_cast<cl_mem>(dstPtr),
                                   3,
                                   rpp::deref(rppHandle),
                                   RPPI_CHN_PACKED,
                                   3);
    }
#elif defined(HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

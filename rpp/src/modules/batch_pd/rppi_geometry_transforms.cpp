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
#include "rppi_geometry_transforms.h"
#include "host_geometry_transforms.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** flip ********************/

RppStatus
rppi_flip_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus resize_host_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType tensorInType,
                             RPPTensorDataType tensorOutType,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensorInType == RPPTensorDataType::U8)
    {
        if (tensorOutType == RPPTensorDataType::U8)
        {
            resize_host_batch<Rpp8u, Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp8u *>(dstPtr),
                                            dstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                            outputFormatToggle,
                                            rpp::deref(rppHandle).GetBatchSize(),
                                            chn_format,
                                            num_of_channels,
                                            rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP16)
        {
            resize_host_batch<Rpp8u, Rpp16f>(static_cast<Rpp8u *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp16f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP32)
        {
            resize_host_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp32f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::I8)
        {
            resize_u8_i8_host_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u *>(srcPtr),
                                                  srcSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                  static_cast<Rpp8s *>(dstPtr),
                                                  dstSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                                  outputFormatToggle,
                                                  rpp::deref(rppHandle).GetBatchSize(),
                                                  chn_format,
                                                  num_of_channels,
                                                  rpp::deref(rppHandle));
        }
    }
    else if (tensorInType == RPPTensorDataType::FP16)
    {
        resize_host_batch<Rpp16f, Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp16f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::FP32)
    {
        resize_host_batch<Rpp32f, Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp32f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::I8)
    {
        resize_host_batch<Rpp8s, Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8s *>(dstPtr),
                                        dstSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
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
rppi_resize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** resize_crop ********************/

RppStatus resize_crop_host_helper(RppiChnFormat chn_format,
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
                                  Rpp32u outputFormatToggle,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        resize_crop_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      dstSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                      xRoiBegin,
                                      xRoiEnd,
                                      yRoiBegin,
                                      yRoiEnd,
                                      outputFormatToggle,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      chn_format,
                                      num_of_channels,
                                      rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        resize_crop_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp16f *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       xRoiBegin,
                                       xRoiEnd,
                                       yRoiBegin,
                                       yRoiEnd,
                                       outputFormatToggle,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       chn_format,
                                       num_of_channels,
                                       rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        resize_crop_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp32f *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       xRoiBegin,
                                       xRoiEnd,
                                       yRoiBegin,
                                       yRoiEnd,
                                       outputFormatToggle,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       chn_format,
                                       num_of_channels,
                                       rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        resize_crop_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8s *>(dstPtr),
                                      dstSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                      xRoiBegin,
                                      xRoiEnd,
                                      yRoiBegin,
                                      yRoiEnd,
                                      outputFormatToggle,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      chn_format,
                                      num_of_channels,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** rotate ********************/

RppStatus rotate_host_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType tensor_type,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
                             Rpp32f *angleDeg,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        rotate_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u *>(dstPtr),
                                 dstSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                 angleDeg,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 outputFormatToggle,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 chn_format,
                                 num_of_channels,
                                 rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        rotate_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp16f *>(dstPtr),
                                  dstSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                  angleDeg,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                  outputFormatToggle,
                                  rpp::deref(rppHandle).GetBatchSize(),
                                  chn_format,
                                  num_of_channels,
                                  rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        rotate_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp32f *>(dstPtr),
                                  dstSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                  angleDeg,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                  outputFormatToggle,
                                  rpp::deref(rppHandle).GetBatchSize(),
                                  chn_format,
                                  num_of_channels,
                                  rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        rotate_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8s *>(dstPtr),
                                 dstSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                 angleDeg,
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
rppi_rotate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** warp_affine ********************/

RppStatus warp_affine_host_helper(RppiChnFormat chn_format,
                                  Rpp32u num_of_channels,
                                  RPPTensorDataType in_tensor_type,
                                  RPPTensorDataType out_tensor_type,
                                  Rpp8u outputFormatToggle,
                                  RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  RppiSize *dstSize,
                                  RppiSize maxDstSize,
                                  Rpp32f *affineMatrix,
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

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            warp_affine_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          affineMatrix,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::FP16)
    {
        if (out_tensor_type == RPPTensorDataType::FP16)
        {
            warp_affine_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp16f *>(dstPtr),
                                           dstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           affineMatrix,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::FP32)
    {
        if (out_tensor_type == RPPTensorDataType::FP32)
        {
            warp_affine_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp32f *>(dstPtr),
                                           dstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           affineMatrix,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::I8)
    {
        if (out_tensor_type == RPPTensorDataType::I8)
        {
            warp_affine_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8s *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          affineMatrix,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_affine_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}

/******************** fisheye ********************/

RppStatus
rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              1,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pln3_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PACKED,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus
rppi_lens_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      1,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PACKED,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** scale ********************/

RppStatus
rppi_scale_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** warp_perspective ********************/

RppStatus
rppi_warp_perspective_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       1,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
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

/******************** flip ********************/

RppStatus
rppi_flip_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus resize_helper(RppiChnFormat chn_format,
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
                        Rpp32u nbatchSize,
                        rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);

#ifdef OCL_COMPILE
    {
        resize_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
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
                resize_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                        static_cast<Rpp8u *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                resize_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp16f *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                resize_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp32f *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                resize_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp8s *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            resize_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                         static_cast<Rpp16f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            resize_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                         static_cast<Rpp32f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            resize_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                         static_cast<Rpp8s *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}

/******************** resize_crop ********************/

RppStatus resize_crop_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType in_tensor_type,
                             RPPTensorDataType out_tensor_type,
                             Rpp8u outputFormatToggle,
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
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        resize_crop_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
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
                resize_crop_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                             static_cast<Rpp8u *>(dstPtr),
                                             rpp::deref(rppHandle),
                                             tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                resize_crop_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp16f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                resize_crop_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp32f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                resize_crop_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp8s *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            resize_crop_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            resize_crop_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            resize_crop_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}

/******************** rotate ********************/

RppStatus rotate_helper(RppiChnFormat chn_format,
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
                        Rpp32f *angleDeg,
                        Rpp32u nbatchSize,
                        rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
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
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_float(angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        rotate_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            rotate_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                    static_cast<Rpp8u *>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            rotate_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                         static_cast<Rpp16f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            rotate_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                         static_cast<Rpp32f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            rotate_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                         static_cast<Rpp8s *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
    }
#endif //BACKEND
    return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}

/******************** warp_affine ********************/

RppStatus warp_affine_helper(RppiChnFormat chn_format,
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
                             Rpp32f *affineMatrix,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);

#ifdef OCL_COMPILE
    {
        warp_affine_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    rpp::deref(rppHandle), affineMatrix,
                                    tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            warp_affine_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                         static_cast<Rpp8u *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         affineMatrix,
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            warp_affine_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            warp_affine_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            warp_affine_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}

/******************** fisheye ********************/

RppStatus
rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         1);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         3);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PACKED,
                         3);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus
rppi_lens_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 1);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PACKED,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** scale ********************/

RppStatus
rppi_scale_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** warp_perspective ********************/

RppStatus
rppi_warp_perspective_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PACKED,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PACKED,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

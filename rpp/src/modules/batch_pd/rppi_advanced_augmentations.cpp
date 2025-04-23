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
#include "rppi_advanced_augmentations.h"
#include "host_advanced_augmentations.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_load_store.hpp"
#include "hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** water ********************/

RppStatus water_host_helper(RppiChnFormat chn_format,
                            Rpp32u num_of_channels,
                            RPPTensorDataType in_tensor_type,
                            RPPTensorDataType out_tensor_type,
                            Rpp8u outputFormatToggle,
                            RppPtr_t srcPtr,
                            RppiSize *srcSize,
                            RppiSize maxSrcSize,
                            RppPtr_t dstPtr,
                            Rpp32f *ampl_x,
                            Rpp32f *ampl_y,
                            Rpp32f *freq_x,
                            Rpp32f *freq_y,
                            Rpp32f *phase_x,
                            Rpp32f *phase_y,
                            Rpp32u nbatchSize,
                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            water_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u *>(dstPtr),
                                    ampl_x,
                                    ampl_y,
                                    freq_x,
                                    freq_y,
                                    phase_x,
                                    phase_y,
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
            water_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp16f *>(dstPtr),
                                     ampl_x,
                                     ampl_y,
                                     freq_x,
                                     freq_y,
                                     phase_x,
                                     phase_y,
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
            water_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp32f *>(dstPtr),
                                     ampl_x,
                                     ampl_y,
                                     freq_x,
                                     freq_y,
                                     phase_x,
                                     phase_y,
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
            water_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8s *>(dstPtr),
                                    ampl_x,
                                    ampl_y,
                                    freq_x,
                                    freq_y,
                                    phase_x,
                                    phase_y,
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
rppi_water_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

/******************** non_linear_blend ********************/

RppStatus non_linear_blend_host_helper(RppiChnFormat chn_format,
                                       Rpp32u num_of_channels,
                                       RPPTensorDataType in_tensor_type,
                                       RPPTensorDataType out_tensor_type,
                                       Rpp8u outputFormatToggle,
                                       RppPtr_t srcPtr1,
                                       RppPtr_t srcPtr2,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32f *std_dev,
                                       Rpp32u nbatchSize,
                                       rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            non_linear_blend_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr1),
                                               static_cast<Rpp8u *>(srcPtr2),
                                               srcSize,
                                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                               static_cast<Rpp8u *>(dstPtr),
                                               std_dev,
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
            non_linear_blend_f16_host_batch(static_cast<Rpp16f *>(srcPtr1),
                                            static_cast<Rpp16f *>(srcPtr2),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp16f *>(dstPtr),
                                            std_dev,
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
            non_linear_blend_f32_host_batch(static_cast<Rpp32f *>(srcPtr1),
                                            static_cast<Rpp32f *>(srcPtr2),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp32f *>(dstPtr),
                                            std_dev,
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
            non_linear_blend_i8_host_batch(static_cast<Rpp8s *>(srcPtr1),
                                           static_cast<Rpp8s *>(srcPtr2),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp8s *>(dstPtr),
                                           std_dev,
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
rppi_non_linear_blend_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

/******************** color_cast ********************/

RppStatus color_cast_host_helper(RppiChnFormat chn_format,
                                 Rpp32u num_of_channels,
                                 RPPTensorDataType in_tensor_type,
                                 RPPTensorDataType out_tensor_type,
                                 Rpp8u outputFormatToggle,
                                 RppPtr_t srcPtr,
                                 RppiSize *srcSize,
                                 RppiSize maxSrcSize,
                                 RppPtr_t dstPtr,
                                 Rpp8u *r,
                                 Rpp8u *g,
                                 Rpp8u *b,
                                 Rpp32f *alpha,
                                 Rpp32u nbatchSize,
                                 rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            color_cast_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         static_cast<Rpp8u *>(dstPtr),
                                         r,
                                         g,
                                         b,
                                         alpha,
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
            color_cast_f16_host_batch(static_cast<Rpp16f *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp16f *>(dstPtr),
                                      r,
                                      g,
                                      b,
                                      alpha,
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
            color_cast_f32_host_batch(static_cast<Rpp32f *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp32f *>(dstPtr),
                                      r,
                                      g,
                                      b,
                                      alpha,
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
            color_cast_i8_host_batch(static_cast<Rpp8s *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp8s *>(dstPtr),
                                     r,
                                     g,
                                     b,
                                     alpha,
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
rppi_color_cast_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}

/******************** erase ********************/

RppStatus erase_host_helper(RppiChnFormat chn_format,
                            Rpp32u num_of_channels,
                            RPPTensorDataType in_tensor_type,
                            RPPTensorDataType out_tensor_type,
                            Rpp8u outputFormatToggle,
                            RppPtr_t srcPtr,
                            RppiSize *srcSize,
                            RppiSize maxSrcSize,
                            RppPtr_t dstPtr,
                            RppPtr_t anchor_box_info,
                            RppPtr_t colors,
                            RppPtr_t box_offset,
                            Rpp32u *num_of_boxes,
                            Rpp32u nbatchSize,
                            rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            erase_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u *>(dstPtr),
                                    static_cast<Rpp32u *>(anchor_box_info),
                                    static_cast<Rpp8u *>(colors),
                                    static_cast<Rpp32u *>(box_offset),
                                    static_cast<Rpp32u *>(num_of_boxes),
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
            erase_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp16f *>(dstPtr),
                                     static_cast<Rpp32u *>(anchor_box_info),
                                     static_cast<Rpp16f *>(colors),
                                     static_cast<Rpp32u *>(box_offset),
                                     static_cast<Rpp32u *>(num_of_boxes),
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
            erase_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp32f *>(dstPtr),
                                     static_cast<Rpp32u *>(anchor_box_info),
                                     static_cast<Rpp32f *>(colors),
                                     static_cast<Rpp32u *>(box_offset),
                                     static_cast<Rpp32u *>(num_of_boxes),
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
            erase_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8s *>(dstPtr),
                                    static_cast<Rpp32u *>(anchor_box_info),
                                    static_cast<Rpp8s *>(colors),
                                    static_cast<Rpp32u *>(box_offset),
                                    static_cast<Rpp32u *>(num_of_boxes),
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
rppi_erase_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}

/******************** crop_and_patch ********************/

RppStatus crop_and_patch_host_helper(RppiChnFormat chn_format,
                                     Rpp32u num_of_channels,
                                     RPPTensorDataType in_tensor_type,
                                     RPPTensorDataType out_tensor_type,
                                     Rpp8u outputFormatToggle,
                                     RppPtr_t srcPtr1,
                                     RppPtr_t srcPtr2,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32u *x11,
                                     Rpp32u *y11,
                                     Rpp32u *x12,
                                     Rpp32u *y12,
                                     Rpp32u *x21,
                                     Rpp32u *y21,
                                     Rpp32u *x22,
                                     Rpp32u *y22,
                                     Rpp32u nbatchSize,
                                     rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            crop_and_patch_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr1),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u *>(srcPtr2),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8u *>(dstPtr),
                                             x11,
                                             y11,
                                             x12,
                                             y12,
                                             x21,
                                             y21,
                                             x22,
                                             y22,
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
            crop_and_patch_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr1),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp16f *>(srcPtr2),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp16f *>(dstPtr),
                                              x11,
                                              y11,
                                              x12,
                                              y12,
                                              x21,
                                              y21,
                                              x22,
                                              y22,
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
            crop_and_patch_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr1),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp32f *>(srcPtr2),
                                              srcSize,
                                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                              static_cast<Rpp32f *>(dstPtr),
                                              x11,
                                              y11,
                                              x12,
                                              y12,
                                              x21,
                                              y21,
                                              x22,
                                              y22,
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
            crop_and_patch_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr1),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8s *>(srcPtr2),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp8s *>(dstPtr),
                                             x11,
                                             y11,
                                             x12,
                                             y12,
                                             x21,
                                             y21,
                                             x22,
                                             y22,
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
rppi_crop_and_patch_u8_pln1_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pln1_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pln1_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pln1_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_u8_pln3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pln3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pln3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pln3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pkd3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pkd3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pkd3_batchPD_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}

/******************** lut ********************/

RppStatus lut_host_helper(RppiChnFormat chn_format,
                          Rpp32u num_of_channels,
                          RPPTensorDataType in_tensor_type,
                          RPPTensorDataType out_tensor_type,
                          Rpp8u outputFormatToggle,
                          RppPtr_t srcPtr,
                          RppiSize *srcSize,
                          RppiSize maxSrcSize,
                          RppPtr_t dstPtr,
                          RppPtr_t lut,
                          Rpp32u nbatchSize,
                          rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            lut_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp8u *>(dstPtr),
                                  static_cast<Rpp8u *>(lut),
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
            lut_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp8s *>(dstPtr),
                                  static_cast<Rpp8s *>(lut),
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
rppi_lut_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}

/******************** glitch ********************/

RppStatus glitch_host_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType in_tensor_type,
                             RPPTensorDataType out_tensor_type,
                             Rpp8u outputFormatToggle,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u *x_offset_r,
                             Rpp32u *y_offset_r,
                             Rpp32u *x_offset_g,
                             Rpp32u *y_offset_g,
                             Rpp32u *x_offset_b,
                             Rpp32u *y_offset_b,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            glitch_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp8u *>(dstPtr),
                                     x_offset_r,
                                     y_offset_r,
                                     x_offset_g,
                                     y_offset_g,
                                     x_offset_b,
                                     y_offset_b,
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
            glitch_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp16f *>(dstPtr),
                                      x_offset_r,
                                      y_offset_r,
                                      x_offset_g,
                                      y_offset_g,
                                      x_offset_b,
                                      y_offset_b,
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
            glitch_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp32f *>(dstPtr),
                                      x_offset_r,
                                      y_offset_r,
                                      x_offset_g,
                                      y_offset_g,
                                      x_offset_b,
                                      y_offset_b,
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
            glitch_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                     srcSize,
                                     rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                     static_cast<Rpp8s *>(dstPtr),
                                     x_offset_r,
                                     y_offset_r,
                                     x_offset_g,
                                     y_offset_g,
                                     x_offset_b,
                                     y_offset_b,
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
rppi_glitch_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** water ********************/

RppStatus water_helper(RppiChnFormat chn_format,
                       Rpp32u num_of_channels,
                       RPPTensorDataType in_tensor_type,
                       RPPTensorDataType out_tensor_type,
                       Rpp8u outputFormatToggle,
                       RppPtr_t srcPtr,
                       RppiSize *srcSize,
                       RppiSize maxSrcSize,
                       RppPtr_t dstPtr,
                       Rpp32f *ampl_x,
                       Rpp32f *ampl_y,
                       Rpp32f *freq_x,
                       Rpp32f *freq_y,
                       Rpp32f *phase_x,
                       Rpp32f *phase_y,
                       Rpp32u nbatchSize,
                       rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_float(ampl_x, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(ampl_y, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(freq_x, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(freq_y, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(phase_x, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(phase_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        water_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       tensor_info);
    }

#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            water_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            water_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                        static_cast<Rpp16f *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            water_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                        static_cast<Rpp32f *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            water_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                        static_cast<Rpp8s *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_water_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}
RppStatus
rppi_water_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

/******************** non_linear_blend ********************/

RppStatus non_linear_blend_helper(RppiChnFormat chn_format,
                                  Rpp32u num_of_channels,
                                  RPPTensorDataType in_tensor_type,
                                  RPPTensorDataType out_tensor_type,
                                  Rpp8u outputFormatToggle,
                                  RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32f *std_dev,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_float(std_dev, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        non_linear_blend_cl_batch(static_cast<cl_mem>(srcPtr1),
        static_cast<cl_mem>(srcPtr2),
        static_cast<cl_mem>(dstPtr),
        rpp::deref(rppHandle),
        tensor_info);
    }

#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            non_linear_blend_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr1),
                                              static_cast<Rpp8u *>(srcPtr2),
                                              static_cast<Rpp8u *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            non_linear_blend_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr1),
                                                   static_cast<Rpp16f *>(srcPtr2),
                                                   static_cast<Rpp16f *>(dstPtr),
                                                   rpp::deref(rppHandle),
                                                   tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            non_linear_blend_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr1),
                                                   static_cast<Rpp32f *>(srcPtr2),
                                                   static_cast<Rpp32f *>(dstPtr),
                                                   rpp::deref(rppHandle),
                                                   tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            non_linear_blend_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr1),
                                                   static_cast<Rpp8s *>(srcPtr2),
                                                   static_cast<Rpp8s *>(dstPtr),
                                                   rpp::deref(rppHandle),
                                                   tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_non_linear_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
RppStatus
rppi_non_linear_blend_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (non_linear_blend_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

/******************** color_cast ********************/

RppStatus color_cast_helper(RppiChnFormat chn_format,
                            Rpp32u num_of_channels,
                            RPPTensorDataType in_tensor_type,
                            RPPTensorDataType out_tensor_type,
                            Rpp8u outputFormatToggle,
                            RppPtr_t srcPtr,
                            RppiSize *srcSize,
                            RppiSize maxSrcSize,
                            RppPtr_t dstPtr,
                            Rpp8u *r,
                            Rpp8u *g,
                            Rpp8u *b,
                            Rpp32f *alpha,
                            Rpp32u nbatchSize,
                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uchar(r, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(g, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar(b, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        color_cast_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            color_cast_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                        static_cast<Rpp8u *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            color_cast_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                             static_cast<Rpp16f *>(dstPtr),
                                             rpp::deref(rppHandle),
                                             tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            color_cast_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                             static_cast<Rpp32f *>(dstPtr),
                                             rpp::deref(rppHandle),
                                             tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            color_cast_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                             static_cast<Rpp8s *>(dstPtr),
                                             rpp::deref(rppHandle),
                                             tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_color_cast_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}
RppStatus
rppi_color_cast_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (color_cast_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, r, g, b, alpha, nbatchSize, rppHandle));
}

/******************** erase ********************/

RppStatus erase_helper(RppiChnFormat chn_format,
                       Rpp32u num_of_channels,
                       RPPTensorDataType in_tensor_type,
                       RPPTensorDataType out_tensor_type,
                       Rpp8u outputFormatToggle,
                       RppPtr_t srcPtr,
                       RppiSize *srcSize,
                       RppiSize maxSrcSize,
                       RppPtr_t dstPtr,
                       RppPtr_t anchor_box_info,
                       RppPtr_t colors,
                       RppPtr_t box_offset,
                       Rpp32u *num_of_boxes,
                       Rpp32u nbatchSize,
                       rppHandle_t rppHandle)
{
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(num_of_boxes, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        erase_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       static_cast<cl_mem>(anchor_box_info),
                       static_cast<cl_mem>(colors),
                       static_cast<cl_mem>(box_offset),
                       rpp::deref(rppHandle),
                       tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            erase_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   static_cast<Rpp32u *>(anchor_box_info),
                                   static_cast<Rpp8u *>(colors),
                                   static_cast<Rpp32u *>(box_offset),
                                   rpp::deref(rppHandle),
                                   tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            erase_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                        static_cast<Rpp16f *>(dstPtr),
                                        static_cast<Rpp32u *>(anchor_box_info),
                                        static_cast<Rpp16f *>(colors),
                                        static_cast<Rpp32u *>(box_offset),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            erase_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                        static_cast<Rpp32f *>(dstPtr),
                                        static_cast<Rpp32u *>(anchor_box_info),
                                        static_cast<Rpp32f *>(colors),
                                        static_cast<Rpp32u *>(box_offset),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            erase_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                        static_cast<Rpp8s *>(dstPtr),
                                        static_cast<Rpp32u *>(anchor_box_info),
                                        static_cast<Rpp8s *>(colors),
                                        static_cast<Rpp32u *>(box_offset),
                                        rpp::deref(rppHandle),
                                        tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_erase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, num_of_boxes, nbatchSize, rppHandle));
}

/******************** crop_and_patch ********************/

RppStatus crop_and_patch_helper(RppiChnFormat chn_format,
                                Rpp32u num_of_channels,
                                RPPTensorDataType in_tensor_type,
                                RPPTensorDataType out_tensor_type,
                                Rpp8u outputFormatToggle,
                                RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u *x11,
                                Rpp32u *y11,
                                Rpp32u *x12,
                                Rpp32u *y12,
                                Rpp32u *x21,
                                Rpp32u *y21,
                                Rpp32u *x22,
                                Rpp32u *y22,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(x11, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y11, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x12, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y12, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x21, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y21, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x22, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y22, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        crop_and_patch_cl_batch(static_cast<cl_mem>(srcPtr1),
                                static_cast<cl_mem>(srcPtr2),
                                static_cast<cl_mem>(dstPtr),
                                rpp::deref(rppHandle),
                                tensor_info);
    }

#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            crop_and_patch_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr1),
                                            static_cast<Rpp8u *>(srcPtr2),
                                            static_cast<Rpp8u *>(dstPtr),
                                            rpp::deref(rppHandle),
                                            tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            crop_and_patch_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr1),
                                                 static_cast<Rpp16f *>(srcPtr2),
                                                 static_cast<Rpp16f *>(dstPtr),
                                                 rpp::deref(rppHandle),
                                                 tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            crop_and_patch_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr1),
                                                 static_cast<Rpp32f *>(srcPtr2),
                                                 static_cast<Rpp32f *>(dstPtr),
                                                 rpp::deref(rppHandle),
                                                 tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            crop_and_patch_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr1),
                                                 static_cast<Rpp8s *>(srcPtr2),
                                                 static_cast<Rpp8s *>(dstPtr),
                                                 rpp::deref(rppHandle),
                                                 tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_crop_and_patch_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_and_patch_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (crop_and_patch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, x11, y11, x12, y12, x21, y21, x22, y22, nbatchSize, rppHandle));
}

/******************** lut ********************/

RppStatus lut_helper(RppiChnFormat chn_format,
                     Rpp32u num_of_channels,
                     RPPTensorDataType in_tensor_type,
                     RPPTensorDataType out_tensor_type,
                     Rpp8u outputFormatToggle,
                     RppPtr_t srcPtr,
                     RppiSize *srcSize,
                     RppiSize maxSrcSize,
                     RppPtr_t dstPtr,
                     RppPtr_t lut,
                     Rpp32u nbatchSize,
                     rppHandle_t rppHandle)
{
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);

#ifdef OCL_COMPILE
    {
        lut_cl_batch(static_cast<cl_mem>(srcPtr),
                     static_cast<cl_mem>(dstPtr),
                     static_cast<cl_mem>(lut),
                     rpp::deref(rppHandle),
                     tensor_info);
    }

#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            lut_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                 static_cast<Rpp8u *>(dstPtr),
                                 static_cast<Rpp8u *>(lut),
                                 rpp::deref(rppHandle),
                                 tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            lut_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                      static_cast<Rpp8s *>(dstPtr),
                                      static_cast<Rpp8s *>(lut),
                                      rpp::deref(rppHandle),
                                      tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_lut_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}
RppStatus
rppi_lut_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (lut_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, lut, nbatchSize, rppHandle));
}

/******************** glitch ********************/

RppStatus glitch_helper(RppiChnFormat chn_format,
                        Rpp32u num_of_channels,
                        RPPTensorDataType in_tensor_type,
                        RPPTensorDataType out_tensor_type,
                        Rpp8u outputFormatToggle,
                        RppPtr_t srcPtr,
                        RppiSize *srcSize,
                        RppiSize maxSrcSize,
                        RppPtr_t dstPtr,
                        Rpp32u *x_offset_r,
                        Rpp32u *y_offset_r,
                        Rpp32u *x_offset_g,
                        Rpp32u *y_offset_g,
                        Rpp32u *x_offset_b,
                        Rpp32u *y_offset_b,
                        Rpp32u nbatchSize,
                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    bool is_padded = true;
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(x_offset_r, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y_offset_r, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x_offset_g, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y_offset_g, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(x_offset_b, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(y_offset_b, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        glitch_cl_batch(static_cast<cl_mem>(srcPtr),
                        static_cast<cl_mem>(dstPtr),
                        rpp::deref(rppHandle),
                        tensor_info);
    }

#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            glitch_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                    static_cast<Rpp8u *>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            glitch_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                         static_cast<Rpp16f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            glitch_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                         static_cast<Rpp32f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            glitch_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                         static_cast<Rpp8s *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_glitch_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}
RppStatus
rppi_glitch_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (glitch_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, x_offset_r, y_offset_r, x_offset_g, y_offset_g, x_offset_b, y_offset_b, nbatchSize, rppHandle));
}

#endif // GPU_SUPPORT

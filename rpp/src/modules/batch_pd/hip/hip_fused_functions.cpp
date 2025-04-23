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

#include "hip_declarations.hpp"
#include "rpp_hip_host_decls.hpp"

/******************** color_twist ********************/

RppStatus
color_twist_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float alpha, float beta, float hue_shift, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "color_twist.cpp", "color_twist_batch", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   plnpkdind,
                                                                                   plnpkdind);

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_twist_batch(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_twist_batch_fp16(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_twist_batch_fp32(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_twist_batch_int8(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** resize_crop_mirror ********************/

RppStatus
resize_crop_mirror_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_mirror_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_mirror_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_mirror_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_mirror_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** crop ********************/

RppStatus
crop_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** crop_mirror_normalize ********************/

RppStatus
crop_mirror_normalize_hip(
    Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize, Rpp32u crop_pox_x, Rpp32u crop_pos_y,
    Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
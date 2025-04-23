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

#include "rpp_cl_common.hpp"
#include "cl_declarations.hpp"
#include<iostream>

/******************** color_twist ********************/

RppStatus
color_twist_cl(cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr, const float alpha, const float beta, const float hue_factor, const float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};
    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "colortwist.cl", "colortwist_pln", vld, vgd, "")(srcPtr,
                                                                                 dstPtr,
                                                                                 alpha,
                                                                                 beta,
                                                                                 hue_factor,
                                                                                 sat,
                                                                                 srcSize.height,
                                                                                 srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "colortwist.cl", "colortwist_pkd", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  alpha,
                                                                                  beta,
                                                                                  hue_factor,
                                                                                  sat,
                                                                                  srcSize.height,
                                                                                  srcSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
color_twist_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,  RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{ max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "colortwist.cl", "colortwist_batch", vld, vgd, "")(srcPtr,
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
                                                                                plnpkdind, plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
color_twist_cl_batch_tensor(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info )
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "colortwist.cl";
    std::string kernel_name = "colortwist_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "",  kernel_file, kernel_name, vld, vgd, "")(srcPtr,
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
                                                                      handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                      in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** crop_mirror_normalize ********************/

RppStatus
crop_mirror_normalize_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiSize dstSize, Rpp32u crop_pox_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "crop_mirror_normalize.cl";
    std::string kernel_name = "crop_mirror_normalize_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                     handle_obj->mem.mgpu.dstSize.height,
                                                                     handle_obj->mem.mgpu.dstSize.width,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                     handle_obj->mem.mgpu.floatArr[2].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[3].floatmem,
                                                                     handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.maxDstSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     handle_obj->mem.mgpu.dstBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** crop ********************/

RppStatus
crop_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width , max_height, handle.GetBatchSize()};
    std::string kernel_file  = "crop.cl";
    std::string kernel_name = "crop_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "",kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                    dstPtr,
                                                                    handle_obj->mem.mgpu.dstSize.height,
                                                                    handle_obj->mem.mgpu.dstSize.width,
                                                                    handle_obj->mem.mgpu.srcSize.width,
                                                                    handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                    handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                    handle_obj->mem.mgpu.maxSrcSize.width,
                                                                    handle_obj->mem.mgpu.maxDstSize.width,
                                                                    handle_obj->mem.mgpu.srcBatchIndex,
                                                                    handle_obj->mem.mgpu.dstBatchIndex,
                                                                    tensor_info._in_channels,
                                                                    handle_obj->mem.mgpu.inc,
                                                                    handle_obj->mem.mgpu.dstInc,
                                                                    in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** resize_crop_mirror ********************/

RppStatus
resize_crop_mirror_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "resize.cl";
    std::string kernel_name = "resize_crop_mirror_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file , kernel_name, vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      handle_obj->mem.mgpu.srcSize.height,
                                                                      handle_obj->mem.mgpu.srcSize.width,
                                                                      handle_obj->mem.mgpu.dstSize.height,
                                                                      handle_obj->mem.mgpu.dstSize.width,
                                                                      handle_obj->mem.mgpu.maxSrcSize.width,
                                                                      handle_obj->mem.mgpu.maxDstSize.width,
                                                                      handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                      handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                      handle_obj->mem.mgpu.uintArr[2].uintmem,
                                                                      handle_obj->mem.mgpu.uintArr[3].uintmem,
                                                                      handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                      handle_obj->mem.mgpu.srcBatchIndex,
                                                                      handle_obj->mem.mgpu.dstBatchIndex,
                                                                      tensor_info._in_channels,
                                                                      handle_obj->mem.mgpu.inc,
                                                                      handle_obj->mem.mgpu.dstInc,
                                                                      in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}
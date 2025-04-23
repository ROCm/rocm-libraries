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

/******************** fisheye ********************/

RppStatus
fisheye_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fisheye.cpp", "fisheye_planar", vld, vgd, "")(srcPtr,
                                                                                 dstPtr,
                                                                                 srcSize.height,
                                                                                 srcSize.width,
                                                                                 channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fisheye.cpp", "fisheye_packed", vld, vgd, "")(srcPtr,
                                                                                 dstPtr,
                                                                                 srcSize.height,
                                                                                 srcSize.width,
                                                                                 channel);
    }

    return RPP_SUCCESS;
}

RppStatus
fisheye_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    hip_exec_fisheye_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus
lens_correction_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float strength,float zoom, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (strength == 0)
        strength = 0.000001;
    float halfWidth = (float)srcSize.width / 2.0;
    float halfHeight = (float)srcSize.height / 2.0;
    float correctionRadius = (float)sqrt((float)srcSize.width * srcSize.width + srcSize.height * srcSize.height) / (float)strength;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pln", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            strength,
                                                                                            zoom,
                                                                                            halfWidth,
                                                                                            halfHeight,
                                                                                            correctionRadius,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pkd", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            strength,
                                                                                            zoom,
                                                                                            halfWidth,
                                                                                            halfHeight,
                                                                                            correctionRadius,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
lens_correction_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    hip_exec_lens_correction_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** flip ********************/

RppStatus
flip_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, uint flipAxis, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_planar", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_planar", vld, vgd, "")(srcPtr,
                                                                                         dstPtr,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel);
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_planar", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_packed", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_packed", vld, vgd, "")(srcPtr,
                                                                                         dstPtr,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel);
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_packed", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
        }
    }

    return RPP_SUCCESS;
}

RppStatus
flip_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_flip_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** scale ********************/

RppStatus
scale_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiSize dstSize, Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    percentage /= 100;
    unsigned int dstheight = (Rpp32s) (percentage * (Rpp32f) srcSize.height);
    unsigned int dstwidth = (Rpp32s) (percentage * (Rpp32f) srcSize.width);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pln", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         dstSize.height,
                                                                         dstSize.width,
                                                                         channel,
                                                                         dstheight,
                                                                         dstwidth);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pkd", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         dstSize.height,
                                                                         dstSize.width,
                                                                         channel,
                                                                         dstheight,
                                                                         dstwidth);

    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
scale_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_scale_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** random_crop_letterbox ********************/

RppStatus
random_crop_letterbox_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;
    unsigned int padding = 10;
    unsigned int type = 1;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_random_crop_letterbox_batch(srcPtr, dstPtr, handle, chnFormat, channel, padding, type, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus
resize_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiSize dstSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pkd", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           dstSize.height,
                                                                           dstSize.width,
                                                                           channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           dstSize.height,
                                                                           dstSize.width,
                                                                           channel);
    }

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;
    unsigned int padding = 0;
    unsigned int type = 0;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "resize.cpp", "resize_crop_batch", vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                              handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                              handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                              handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                              channel,
                                                                              handle.GetInitHandle()->mem.mgpu.inc,
                                                                              handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                              padding,
                                                                              type,
                                                                              plnpkdind,
                                                                              plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_fp16(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_fp32(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_int8(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** resize_crop ********************/

RppStatus
resize_crop_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned int type = 0, padding = 0;
    unsigned int width,height;
    width = dstSize.width;
    height = dstSize.height;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pkd", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                height,
                                                                                width,
                                                                                x1,
                                                                                x2,
                                                                                y1,
                                                                                y2,
                                                                                padding,
                                                                                type,
                                                                                channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pln", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                height,
                                                                                width,
                                                                                x1,
                                                                                x2,
                                                                                y1,
                                                                                y2,
                                                                                padding,
                                                                                type,
                                                                                channel);
    }

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    unsigned int padding = 10;
    unsigned int type = 1;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "resize.cpp", "resize_crop_batch", vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                              handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                              handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                              handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                              handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                                                                              handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                              handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                              channel,
                                                                              handle.GetInitHandle()->mem.mgpu.inc,
                                                                              handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                              padding,
                                                                              type,
                                                                              plnpkdind,
                                                                              plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_fp16(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_fp32(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_batch_int8(srcPtr, dstPtr, handle, tensor_info, padding, type, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** rotate ********************/

RppStatus
rotate_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiSize dstSize, float angleDeg, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pkd", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           angleDeg,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           dstSize.height,
                                                                           dstSize.width,
                                                                           channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pln", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           angleDeg,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           dstSize.height,
                                                                           dstSize.width,
                                                                           channel);
    }

    return RPP_SUCCESS;
}

RppStatus
rotate_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "rotate.cpp", "rotate_batch", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                         handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                         handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                         handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                         channel,
                                                                         handle.GetInitHandle()->mem.mgpu.inc,
                                                                         handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                         plnpkdind,
                                                                         plnpkdind);

    return RPP_SUCCESS;
}

RppStatus
rotate_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_rotate_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
rotate_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_rotate_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
rotate_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_rotate_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
rotate_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_rotate_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** warp_affine ********************/

RppStatus
warp_affine_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiSize dstSize, float *affine, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float affine_inv[6];
    float det;
    det = (affine[0] * affine [4])  - (affine[1] * affine[3]);
    affine_inv[0] = affine[4]/ det;
    affine_inv[1] = (- 1 * affine[1])/ det;
    affine_inv[2] = -1 * affine[2];
    affine_inv[3] = (-1 * affine[3]) /det ;
    affine_inv[4] = affine[0]/det;
    affine_inv[5] = -1 * affine[5];
    float *affine_matrix;
    Rpp32u* affine_array;
    hipMalloc(&affine_array, sizeof(float)*6);
    hipMemcpy(affine_array,affine_inv,sizeof(float)*6,hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pln", vld, vgd, "")(srcPtr,
                                                                                     dstPtr,
                                                                                     affine_array,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     dstSize.height,
                                                                                     dstSize.width,
                                                                                     channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pkd", vld, vgd, "")(srcPtr,
                                                                                     dstPtr,
                                                                                     affine_array,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     dstSize.height,
                                                                                     dstSize.width,
                                                                                     channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle,Rpp32f *affine, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u nBatchSize = handle.GetBatchSize();
    Rpp32f *hip_affine;
    hipMalloc(&hip_affine, nBatchSize * 6 * sizeof(Rpp32f));
    hipMemcpy(hip_affine, affine, nBatchSize * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_batch", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
                                                                                   hip_affine,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                   handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                                   channel,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                                   plnpkdind);

    hipFree(&hip_affine);

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f *hipAffine;
    hipMalloc(&hipAffine, handle.GetBatchSize() * 6 * sizeof(Rpp32f));
    hipMemcpy(hipAffine, affine, handle.GetBatchSize() * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_warp_affine_batch(srcPtr, dstPtr, handle, hipAffine, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    hipFree(hipAffine);

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f *hipAffine;
    hipMalloc(&hipAffine, handle.GetBatchSize() * 6 * sizeof(Rpp32f));
    hipMemcpy(hipAffine, affine, handle.GetBatchSize() * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_warp_affine_batch_fp16(srcPtr, dstPtr, handle, hipAffine, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f *hipAffine;
    hipMalloc(&hipAffine, handle.GetBatchSize() * 6 * sizeof(Rpp32f));
    hipMemcpy(hipAffine, affine, handle.GetBatchSize() * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_warp_affine_batch_fp32(srcPtr, dstPtr, handle, hipAffine, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f *hipAffine;
    hipMalloc(&hipAffine, handle.GetBatchSize() * 6 * sizeof(Rpp32f));
    hipMemcpy(hipAffine, affine, handle.GetBatchSize() * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_warp_affine_batch_int8(srcPtr, dstPtr, handle, hipAffine, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** warp_perspective ********************/

RppStatus
warp_perspective_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiSize dstSize, float *perspective, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float perspective_inv[9];
    float det;
    det = (perspective[0] * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) - (perspective[1] * ((perspective[3] * perspective[8]) - (perspective[5] * perspective[6]))) + (perspective[2] * ((perspective[3] * perspective[7]) - (perspective[4] * perspective[6])));
    perspective_inv[0] = (1 * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) / det;
    perspective_inv[1] = (-1 * ((perspective[1] * perspective[8]) - (perspective[7] * perspective[2]))) / det;
    perspective_inv[2] = (1 * ((perspective[1] * perspective[5]) - (perspective[4] * perspective[2]))) / det;
    perspective_inv[3] = (-1 * ((perspective[3] * perspective[8]) - (perspective[6] * perspective[5]))) / det;
    perspective_inv[4] = (1 * ((perspective[0] * perspective[8]) - (perspective[6] * perspective[2]))) / det;
    perspective_inv[5] = (-1 * ((perspective[0] * perspective[5]) - (perspective[3] * perspective[2]))) / det;
    perspective_inv[6] = (1 * ((perspective[3] * perspective[7]) - (perspective[6] * perspective[4]))) / det;
    perspective_inv[7] = (-1 * ((perspective[0] * perspective[7]) - (perspective[6] * perspective[1]))) / det;
    perspective_inv[8] = (1 * ((perspective[0] * perspective[4]) - (perspective[3] * perspective[1]))) / det;
    float *perspective_matrix;
    Rpp32f* perspective_array;
    hipMalloc(&perspective_array, sizeof(float) * 9);
    hipMemcpy(perspective_array, perspective_inv,sizeof(float) * 9, hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                                               dstPtr,
                                                                                               perspective_array,
                                                                                               srcSize.height,
                                                                                               srcSize.width,
                                                                                               dstSize.height,
                                                                                               dstSize.width,
                                                                                               channel);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                                               dstPtr,
                                                                                               perspective_array,
                                                                                               srcSize.height,
                                                                                               srcSize.width,
                                                                                               dstSize.height,
                                                                                               dstSize.width,
                                                                                               channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
warp_perspective_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *perspective, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f* hipPerspective;
    hipMalloc(&hipPerspective, sizeof(float) * 9 * handle.GetBatchSize());
    hipMemcpy(hipPerspective, perspective, sizeof(float) * 9 * handle.GetBatchSize(), hipMemcpyHostToDevice);

    hip_exec_warp_perspective_batch(srcPtr, dstPtr, handle, hipPerspective, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
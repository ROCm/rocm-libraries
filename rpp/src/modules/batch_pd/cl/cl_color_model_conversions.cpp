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

/******************** color_temperature ********************/

RppStatus
color_temperature_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32s adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "color_temperature.cl", "temperature_planar", vld, vgd, "")(srcPtr,
                                                                                             dstPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel,
                                                                                             adjustmentValue);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "color_temperature.cl", "temperature_packed", vld, vgd, "")(srcPtr,
                                                                                             dstPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel,
                                                                                             adjustmentValue);
    }

    return RPP_SUCCESS;
}

RppStatus
color_temperature_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "color_temperature.cl", "color_temperature_batch", vld, vgd, "")(srcPtr,
                                                                                              dstPtr,
                                                                                              handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                              handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                              handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                              handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                              channel,
                                                                                              handle.GetInitHandle()->mem.mgpu.inc,
                                                                                              plnpkdind);

    return RPP_SUCCESS;
}

/******************** vignette ********************/

RppStatus
vignette_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "vignette.cl", "vignette_pln", vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              srcSize.height,
                                                                              srcSize.width,
                                                                              channel,
                                                                              stdDev);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "vignette.cl", "vignette_pkd", vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              srcSize.height,
                                                                              srcSize.width,
                                                                              channel,
                                                                              stdDev);
    }

    return RPP_SUCCESS;
}

RppStatus
vignette_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "vignette.cl", "vignette_batch", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                            channel,
                                                                            handle.GetInitHandle()->mem.mgpu.inc,
                                                                            plnpkdind);

    return RPP_SUCCESS;
}

/******************** channel_extract ********************/

RppStatus
channel_extract_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u extractChannelNumber, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "channel_extract.cl", "channel_extract_pln", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            extractChannelNumber);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "channel_extract.cl", "channel_extract_pkd", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            extractChannelNumber);
    }

    return RPP_SUCCESS;
}

RppStatus
channel_extract_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "channel_extract.cl", "channel_extract_batch", vld, vgd, "")(srcPtr,
                                                                                          dstPtr,
                                                                                          handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                          channel,
                                                                                          handle.GetInitHandle()->mem.mgpu.inc,
                                                                                          plnpkdind);

    return RPP_SUCCESS;
}

/******************** channel_combine ********************/

RppStatus
channel_combine_cl(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem srcPtr3, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "channel_combine.cl", "channel_combine_pln", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            srcPtr3,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "channel_combine.cl", "channel_combine_pkd", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            srcPtr3,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);
    }

    return RPP_SUCCESS;
}

RppStatus
channel_combine_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem srcPtr3, cl_mem dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "channel_combine.cl", "channel_combine_batch", vld, vgd, "")(srcPtr1,
                                                                                          srcPtr2,
                                                                                          srcPtr3,
                                                                                          dstPtr,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                          channel,
                                                                                          handle.GetInitHandle()->mem.mgpu.inc,
                                                                                          plnpkdind);

    return RPP_SUCCESS;
}

/******************** hueRGB ********************/

RppStatus
hueRGB_cl(cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr, float hue_factor, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float sat = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};
    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cl", "huergb_pln", vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      hue_factor,
                                                                      sat,
                                                                      srcSize.height,
                                                                      srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cl", "huergb_pkd", vld, vgd, "")(srcPtr,
                                                                       dstPtr,
                                                                       hue_factor,
                                                                       sat,
                                                                       srcSize.height,
                                                                       srcSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
hueRGB_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "hue.cl", "hue_batch", vld, vgd, "")(srcPtr,
                                                                  dstPtr,
                                                                  handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                  handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                  handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                  handle.GetInitHandle()->mem.mgpu.inc,
                                                                  plnpkdind);

    return RPP_SUCCESS;
}

/******************** saturationRGB ********************/

RppStatus
saturationRGB_cl(cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float hue_factor = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cl", "huergb_pkd", vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      hue_factor,
                                                                      sat,
                                                                      srcSize.height,
                                                                      srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cl", "huergb_pkd", vld, vgd, "")(srcPtr,
                                                                       dstPtr,
                                                                       hue_factor,
                                                                       sat,
                                                                       srcSize.height,
                                                                       srcSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
saturationRGB_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "hue.cl", "saturation_batch", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                         handle.GetInitHandle()->mem.mgpu.inc,
                                                                         plnpkdind);

    return RPP_SUCCESS;
}

/******************** color_convert ********************/

RppStatus
color_convert_cl(cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,  RppiColorConvertMode convert_mode, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned int plnpkdind, inc;
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        plnpkdind = 1;
        inc = srcSize.height * srcSize.width;
    }
    else
    {
        plnpkdind = 3;
        inc = 1;
    }
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};
    if (convert_mode == RGB_HSV)
    {
       handle.AddKernel("", "", "hue.cl", "convert_single_rgb_hsv", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  inc,
                                                                                  plnpkdind);
    }
    else if (convert_mode == HSV_RGB)
    {
        handle.AddKernel("", "", "hue.cl", "convert_single_hsv_rgb", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   inc,
                                                                                   plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus
color_convert_cl_batch(cl_mem srcPtr, cl_mem dstPtr,  RppiColorConvertMode convert_mode, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle){
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    std::string kernel_name ;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    if (convert_mode == RGB_HSV)
        kernel_name = "convert_batch_rgb_hsv";
    if (convert_mode == HSV_RGB)
        kernel_name = "convert_batch_hsv_rgb";
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "hue.cl", kernel_name, vld, vgd, "")(srcPtr,
                                                                  dstPtr,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                  handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                  handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                  handle.GetInitHandle()->mem.mgpu.inc,
                                                                  plnpkdind);

    return RPP_SUCCESS;
}

/******************** look_up_table ********************/

RppStatus
look_up_table_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,Rpp8u* lutPtr,
 RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem clLutPtr = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp8u)*256*channel, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), clLutPtr, CL_TRUE, 0, sizeof(Rpp8u)*256*channel, lutPtr, 0, NULL, NULL);
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "look_up_table.cl", "look_up_table_pln", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        clLutPtr,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "look_up_table.cl", "look_up_table_pkd", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        clLutPtr,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        channel);
    }

    return RPP_SUCCESS;
}

RppStatus
look_up_table_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp8u* lutPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem clLutPtr = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp8u)*256*channel*handle.GetBatchSize(), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), clLutPtr, CL_TRUE, 0, sizeof(Rpp8u)*256*channel*handle.GetBatchSize(), lutPtr, 0, NULL, NULL);
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "look_up_table.cl", "look_up_table_batch", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      clLutPtr,
                                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                      handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                      handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                      channel,
                                                                                      handle.GetInitHandle()->mem.mgpu.inc,
                                                                                      plnpkdind);

    return RPP_SUCCESS;
}

/******************** tensor_look_up_table ********************/

RppStatus
tensor_look_up_table_cl(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr, cl_mem dstPtr, Rpp8u* lutPtr, rpp::Handle& handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem clLutPtr = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp8u)*256, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), clLutPtr, CL_TRUE, 0, sizeof(Rpp8u)*256, lutPtr, 0, NULL, NULL);

    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for(int i = 2 ; i < tensorDimension ; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }

    unsigned int dim1,dim2,dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cl", "tensor_look_up_table", vld, vgd, "")(tensorDimension,
                                                                                srcPtr,
                                                                                dstPtr,
                                                                                dim1,
                                                                                dim2,
                                                                                dim3,
                                                                                clLutPtr);

    return RPP_SUCCESS;
}
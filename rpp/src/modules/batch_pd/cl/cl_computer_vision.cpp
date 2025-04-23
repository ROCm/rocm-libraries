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

/******************** data_object_copy ********************/

RppStatus
data_object_copy_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    return RPP_SUCCESS;
}

RppStatus
data_object_copy_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle,
                          RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned long buffer_size = 0;
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        buffer_size += handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] * channel;
    }
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, buffer_size * sizeof(unsigned char), 0, NULL, NULL);
    return RPP_SUCCESS;
}

/******************** local_binary_pattern ********************/

RppStatus
local_binary_pattern_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

        handle.AddKernel("", "", "local_binary_pattern.cl", "local_binary_pattern_pkd", vld, vgd, "")(srcPtr,
                                                                                                      dstPtr,
                                                                                                      srcSize.height,
                                                                                                      srcSize.width,
                                                                                                      channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "local_binary_pattern.cl", "local_binary_pattern_pln", vld, vgd, "")(srcPtr,
                                                                                                      dstPtr,
                                                                                                      srcSize.height,
                                                                                                      srcSize.width,
                                                                                                      channel);
    }

    return RPP_SUCCESS;
}

RppStatus
local_binary_pattern_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "local_binary_pattern.cl", "local_binary_pattern_batch", vld, vgd, "")(srcPtr,
                                                                                                    dstPtr,
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

/******************** gaussian_image_pyramid ********************/

RppStatus
gaussian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);

    if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cl", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
                                                                                                          dstPtr,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel,
                                                                                                          kernel,
                                                                                                          kernelSize,
                                                                                                          kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cl", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
                                                                                                          dstPtr,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel,
                                                                                                          kernel,
                                                                                                          kernelSize,
                                                                                                          kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
gaussian_image_pyramid_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "gaussian_image_pyramid.cl", "gaussian_image_pyramid_batch", vld, vgd, "")(srcPtr,
                                                                                                        dstPtr,
                                                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
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

/******************** control_flow ********************/

RppStatus
control_flow_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    switch (type)
    {
    case 1:
        handle.AddKernel("", "", "bitwise_AND.cl", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
        break;
    case 2:
        handle.AddKernel("", "", "inclusive_OR.cl", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                                  srcPtr2,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
        break;
    case 3:
        handle.AddKernel("", "", "exclusive_OR.cl", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                                  srcPtr2,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
        break;
    case 4:
        handle.AddKernel("", "", "add.cl", "add", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);
        break;
    case 5:
        handle.AddKernel("", "", "subtract.cl", "subtract", vld, vgd, "")(srcPtr1,
                                                                          srcPtr2,
                                                                          dstPtr,
                                                                          srcSize.height,
                                                                          srcSize.width,
                                                                          channel);
        break;
    case 6:
        handle.AddKernel("", "", "multiply.cl", "multiply", vld, vgd, "")(srcPtr1,
                                                                          srcPtr2,
                                                                          dstPtr,
                                                                          srcSize.height,
                                                                          srcSize.width,
                                                                          channel);
        break;
    case 7:
        handle.AddKernel("", "", "min.cl", "min", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);
        break;
    case 8:
        handle.AddKernel("", "", "max.cl", "max", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);
        break;
    }
    return RPP_SUCCESS;
}

RppStatus
control_flow_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, Rpp32u type, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    switch (type)
    {
    case 1:
        handle.AddKernel("", "", "bitwise_AND.cl", "bitwise_AND_batch", vld, vgd, "")(srcPtr1,
                                                                                      srcPtr2,
                                                                                      dstPtr,
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
        break;
    case 2:
        handle.AddKernel("", "", "inclusive_OR.cl", "inclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                        srcPtr2,
                                                                                        dstPtr,
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
        break;
    case 3:
        handle.AddKernel("", "", "exclusive_OR.cl", "exclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                        srcPtr2,
                                                                                        dstPtr,
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
        break;
    case 4:
        handle.AddKernel("", "", "add.cl", "add_batch", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
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
        break;
    case 5:
        handle.AddKernel("", "", "subtract.cl", "subtract_batch", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
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
        break;
    case 6:
        handle.AddKernel("", "", "multiply.cl", "multiply_batch", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
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
        break;
    case 7:
        handle.AddKernel("", "", "min.cl", "min_batch", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
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
        break;
    case 8:
        handle.AddKernel("", "", "max.cl", "max_batch", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
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
        break;
    }

    return RPP_SUCCESS;
}

/******************** convert_bit_depth ********************/

RppStatus
convert_bit_depth_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    if (type == 1)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_u8s8", vld, vgd, "")(srcPtr,
                                                                                                 dstPtr,
                                                                                                 srcSize.height,
                                                                                                 srcSize.width,
                                                                                                 channel);
    }
    else if (type == 2)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_u8u16", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_u8s16", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
    }
    return RPP_SUCCESS;
}

RppStatus
convert_bit_depth_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp32u type, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)

{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    if (type == 1)
    {
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_batch_u8s8", vld, vgd, "")(srcPtr,
                                                                                                       dstPtr,
                                                                                                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                                       channel,
                                                                                                       handle.GetInitHandle()->mem.mgpu.inc,
                                                                                                       plnpkdind);
    }
    else if (type == 2)
    {
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_batch_u8u16", vld, vgd, "")(srcPtr,
                                                                                                        dstPtr,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                                        channel,
                                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                                        plnpkdind);
    }
    else
    {
        handle.AddKernel("", "", "convert_bit_depth.cl", "convert_bit_depth_batch_u8s16", vld, vgd, "")(srcPtr,
                                                                                                        dstPtr,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                                        channel,
                                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                                        plnpkdind);
    }

    return RPP_SUCCESS;
}

/******************** laplacian_image_pyramid ********************/

RppStatus
laplacian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, srcSize.height * srcSize.width * channel * sizeof(Rpp8u), NULL, NULL);

    if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cl", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
                                                                                                          srcPtr1,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel,
                                                                                                          kernel,
                                                                                                          kernelSize,
                                                                                                          kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cl", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
                                                                                                          srcPtr1,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel,
                                                                                                          kernel,
                                                                                                          kernelSize,
                                                                                                          kernelSize);
    }
    if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "laplacian_image_pyramid.cl", "laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
                                                                                                            dstPtr,
                                                                                                            srcSize.height,
                                                                                                            srcSize.width,
                                                                                                            channel,
                                                                                                            kernel,
                                                                                                            kernelSize,
                                                                                                            kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "laplacian_image_pyramid.cl", "laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
                                                                                                            dstPtr,
                                                                                                            srcSize.height,
                                                                                                            srcSize.width,
                                                                                                            channel,
                                                                                                            kernel,
                                                                                                            kernelSize,
                                                                                                            kernelSize);
    }

    return RPP_SUCCESS;
}
RppStatus
laplacian_image_pyramid_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)

{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned int maxHeight, maxWidth, maxKernelSize;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if (maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    unsigned long batchIndex = 0;

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, maxHeight * maxWidth * channel * sizeof(Rpp8u), NULL, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, maxKernelSize * maxKernelSize * sizeof(Rpp32f), NULL, NULL);

    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
        if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i], handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cl", "gaussian_image_pyramid_pkd_batch", vld, vgd, "")(srcPtr,
                                                                                                                     srcPtr1,
                                                                                                                     maxHeight,
                                                                                                                     maxWidth,
                                                                                                                     channel,
                                                                                                                     kernel,
                                                                                                                     handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                     handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                     batchIndex);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{maxWidth, maxHeight, channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cl", "gaussian_image_pyramid_pln_batch", vld, vgd, "")(srcPtr,
                                                                                                                     srcPtr1,
                                                                                                                     maxHeight,
                                                                                                                     maxWidth,
                                                                                                                     channel,
                                                                                                                     kernel,
                                                                                                                     handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                     handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                     batchIndex);
        }

        if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{maxWidth, maxHeight, channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cl", "laplacian_image_pyramid_pkd_batch", vld, vgd, "")(srcPtr1,
                                                                                                                      dstPtr,
                                                                                                                      maxHeight,
                                                                                                                      maxWidth,
                                                                                                                      channel,
                                                                                                                      kernel,
                                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                      batchIndex);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{maxWidth, maxHeight, channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cl", "laplacian_image_pyramid_pln_batch", vld, vgd, "")(srcPtr1,
                                                                                                                      dstPtr,
                                                                                                                      maxHeight,
                                                                                                                      maxWidth,
                                                                                                                      channel,
                                                                                                                      kernel,
                                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                      batchIndex);
        }
        batchIndex += maxHeight * maxWidth * channel;
    }
    clReleaseMemObject(srcPtr1);
    clReleaseMemObject(kernel);

    return RPP_SUCCESS;
}

/******************** canny_edge_detector ********************/

RppStatus
canny_edge_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp8u minThreshold, Rpp8u maxThreshold, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem gsout = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem tempDest2 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    cl_mem sobelX = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem sobelY = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                       gsin,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                       gsin,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
        }
    }

    unsigned int sobelType = 2;
    unsigned int sobelTypeX = 0;
    unsigned int sobelTypeY = 1;
    unsigned int newChannel = 1;

    if (channel == 1)
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        tempDest1,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelType);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                        tempDest1,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelType);
    }
    if (channel == 1)
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        sobelX,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeX);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelX,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeX);
    }
    if (channel == 1)
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        sobelY,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeY);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelY,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeY);
    }
    handle.AddKernel("", "", "canny_edge_detector.cl", "ced_non_max_suppression", vld, vgd, "")(tempDest1,
                                                                                                sobelX,
                                                                                                sobelY,
                                                                                                tempDest2,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                newChannel,
                                                                                                minThreshold,
                                                                                                maxThreshold);
    if (channel == 1)
    {
        handle.AddKernel("", "", "canny_edge_detector.cl", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       newChannel,
                                                                                       minThreshold,
                                                                                       maxThreshold);
    }
    else
    {
        handle.AddKernel("", "", "canny_edge_detector.cl", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                       gsout,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       newChannel,
                                                                                       minThreshold,
                                                                                       maxThreshold);
    }
    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln1_to_pkd3", vld, vgd, "")(gsout,
                                                                                                       dstPtr,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln1_to_pln3", vld, vgd, "")(gsout,
                                                                                                       dstPtr,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
        }
    }
    clReleaseMemObject(gsin);
    clReleaseMemObject(gsout);

    clReleaseMemObject(tempDest1);
    clReleaseMemObject(tempDest2);

    clReleaseMemObject(sobelX);
    clReleaseMemObject(sobelY);

    return RPP_SUCCESS;
}

RppStatus
canny_edge_detector_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp32u imageDim = maxHeight * maxWidth;

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);
    cl_mem gsout = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);

    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);
    cl_mem tempDest2 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);

    cl_mem sobelX = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);
    cl_mem sobelY = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim, NULL, NULL);

    unsigned long batchIndex = 0;
    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * imageDim * channel, NULL, NULL);

    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * imageDim * channel, 0, NULL, NULL);
        size_t gDim3[3];
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pkd3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                           gsin,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           channel);
            }
            else
            {
                handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                           gsin,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           channel);
            }
        }
        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;
        unsigned int newChannel = 1;
        if (channel == 1)
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                            tempDest1,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelType);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                            tempDest1,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelType);
        }
        if (channel == 1)
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                            sobelX,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelTypeX);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                            sobelX,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelTypeX);
        }
        if (channel == 1)
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                            sobelY,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelTypeY);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(gsin,
                                                                            sobelY,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            sobelTypeY);
        }

        handle.AddKernel("", "", "canny_edge_detector.cl", "ced_non_max_suppression", vld, vgd, "")(tempDest1,
                                                                                                    sobelX,
                                                                                                    sobelY,
                                                                                                    tempDest2,
                                                                                                    maxHeight,
                                                                                                    maxWidth,
                                                                                                    newChannel,
                                                                                                    handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                                    handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);

        if (channel == 1)
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                           dstPtr1,
                                                                                           maxHeight,
                                                                                           maxWidth,
                                                                                           newChannel,
                                                                                           handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                           handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cl", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                           gsout,
                                                                                           maxHeight,
                                                                                           maxWidth,
                                                                                           newChannel,
                                                                                           handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                           handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);
        }

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln1_to_pkd3", vld, vgd, "")(gsout,
                                                                                                           dstPtr1,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           channel);
            }
            else
            {
                handle.AddKernel("", "", "canny_edge_detector.cl", "canny_ced_pln1_to_pln3", vld, vgd, "")(gsout,
                                                                                                           dstPtr1,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           channel);
            }
        }
        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * imageDim * channel, 0, NULL, NULL);
        batchIndex += imageDim * channel;
    }
    return RPP_SUCCESS;
}

/******************** harris_corner_detector ********************/

RppStatus
harris_corner_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u gaussianKernelSize, Rpp32f stdDev, Rpp32u kernelSize, Rpp32f kValue, Rpp32f threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    /* SETTING UP */

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    cl_mem dstFloat = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(float) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem nonMaxDstFloat = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(float) * srcSize.height * srcSize.width, NULL, NULL);

    cl_mem sobelX = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem sobelY = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* RGB to GREY SCALE */

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                  gsin,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                  gsin,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
    }

    unsigned int newChannel = 1;

    /* GAUSSIAN FILTER */

    Rpp32f *kernelMain = (Rpp32f *)calloc(gaussianKernelSize * gaussianKernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, gaussianKernelSize);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);

    if (channel == 1)
    {
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(srcPtr,
                                                                                     tempDest1,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     newChannel,
                                                                                     kernel,
                                                                                     gaussianKernelSize,
                                                                                     gaussianKernelSize);
    }
    else
    {
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                     tempDest1,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     newChannel,
                                                                                     kernel,
                                                                                     gaussianKernelSize,
                                                                                     gaussianKernelSize);
    }

    unsigned int sobelTypeX = 0;
    unsigned int sobelTypeY = 1;
    handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                    sobelX,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    newChannel,
                                                                    sobelTypeX);
    handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                    sobelY,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    newChannel,
                                                                    sobelTypeY);

    /* HARRIS CORNER STRENGTH MATRIX */

    handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_strength", vld, vgd, "")(sobelX,
                                                                                                           sobelY,
                                                                                                           dstFloat,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           newChannel,
                                                                                                           kernelSize,
                                                                                                           kValue,
                                                                                                           threshold);

    /* NON-MAX SUPRESSION */

    handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_nonmax_supression", vld, vgd, "")(dstFloat,
                                                                                                                    nonMaxDstFloat,
                                                                                                                    srcSize.height,
                                                                                                                    srcSize.width,
                                                                                                                    newChannel,
                                                                                                                    nonmaxKernelSize);
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                          nonMaxDstFloat,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel);
    }
    else
    {
        handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                          nonMaxDstFloat,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel);
    }
    return RPP_SUCCESS;
}

RppStatus
harris_corner_detector_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle,
                                RppiChnFormat chnFormat, unsigned int channel)
{
    /* SETTING UP */

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    unsigned int maxHeight, maxWidth, maxKernelSize;
    unsigned long ioBufferSize = 0, singleImageSize = 0;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if (maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i];
    }

    ioBufferSize = maxHeight * maxWidth * channel * handle.GetBatchSize();
    singleImageSize = maxHeight * maxWidth * channel;

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, maxKernelSize * maxKernelSize * sizeof(Rpp32f), NULL, NULL);

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);

    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);

    cl_mem sobelX = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);
    cl_mem sobelY = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);

    cl_mem dstFloat = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(float) * maxHeight * maxWidth, NULL, NULL);
    cl_mem nonMaxDstFloat = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(float) * maxHeight * maxWidth, NULL, NULL);

    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * ioBufferSize, 0, NULL, NULL);

    unsigned long batchIndex = 0;
    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * singleImageSize, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * singleImageSize, NULL, NULL);

    size_t gDim3[3];

    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * singleImageSize, 0, NULL, NULL);
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                      gsin,
                                                                                                      maxHeight,
                                                                                                      maxWidth,
                                                                                                      channel);
            }
            else
            {
                handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                      gsin,
                                                                                                      maxHeight,
                                                                                                      maxWidth,
                                                                                                      channel);
            }
        }

        unsigned int newChannel = 1;

        /* GAUSSIAN FILTER */

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);

        if (channel == 1)
        {
            handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(srcPtr1,
                                                                                         tempDest1,
                                                                                         maxHeight,
                                                                                         maxWidth,
                                                                                         newChannel,
                                                                                         kernel,
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                         tempDest1,
                                                                                         maxHeight,
                                                                                         maxWidth,
                                                                                         newChannel,
                                                                                         kernel,
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }

        /* SOBEL X and Y */

        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                        sobelX,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeX);
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                        sobelY,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeY);

        /* HARRIS CORNER STRENGTH MATRIX */

        handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_strength", vld, vgd, "")(sobelX,
                                                                                                               sobelY,
                                                                                                               dstFloat,
                                                                                                               maxHeight,
                                                                                                               maxWidth,
                                                                                                               newChannel,
                                                                                                               handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[3].floatmem[i],
                                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[4].floatmem[i]);

        /* NON-MAX SUPRESSION */

        handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_nonmax_supression", vld, vgd, "")(dstFloat,
                                                                                                                        nonMaxDstFloat,
                                                                                                                        maxHeight,
                                                                                                                        maxWidth,
                                                                                                                        newChannel,
                                                                                                                        handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]);

        clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, dstPtr1, 0, 0, sizeof(unsigned char) * singleImageSize, 0, NULL, NULL);

        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr1,
                                                                                                              nonMaxDstFloat,
                                                                                                              maxHeight,
                                                                                                              maxWidth,
                                                                                                              channel);
        }
        else
        {
            handle.AddKernel("", "", "harris_corner_detector.cl", "harris_corner_detector_pln", vld, vgd, "")(dstPtr1,
                                                                                                              nonMaxDstFloat,
                                                                                                              maxHeight,
                                                                                                              maxWidth,
                                                                                                              channel);
        }

        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * singleImageSize, 0, NULL, NULL);
        batchIndex += maxHeight * maxWidth * channel;
    }
    return RPP_SUCCESS;
}

/******************** fast_corner_detector ********************/

RppStatus
fast_corner_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u numOfPixels, Rpp8u threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);
    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize.height * srcSize.width, NULL, NULL);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* RGB to GS */

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                  gsin,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                  gsin,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
    }

    /* FAST CORNER IMPLEMENTATION */

    unsigned int newChannel = 1;

    if (channel == 1)
    {
        handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                  tempDest1,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  newChannel,
                                                                                                  threshold,
                                                                                                  numOfPixels);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                  tempDest1,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  newChannel,
                                                                                                  threshold,
                                                                                                  numOfPixels);
    }

    /* NON MAX SUPRESSION */

    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                          dstPtr,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          newChannel,
                                                                                                          nonmaxKernelSize);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                          dstPtr,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          newChannel,
                                                                                                          nonmaxKernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
fast_corner_detector_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);
    cl_mem tempDest1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth, NULL, NULL);

    size_t gDim3[3];

    size_t batchIndex = 0;

    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);

        /* RGB to GS */

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                      gsin,
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                      channel);
            }
            else
            {
                handle.AddKernel("", "", "fast_corner_detector.cl", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                      gsin,
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                      channel);
            }
        }

        /* FAST CORNER IMPLEMENTATION */

        unsigned int newChannel = 1;
        if (channel == 1)
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                      tempDest1,
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                      newChannel,
                                                                                                      handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                      tempDest1,
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                      newChannel,
                                                                                                      handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                      handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }

        /* NON MAX SUPRESSION */

        clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, dstPtr1, 0, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                              dstPtr,
                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                              newChannel,
                                                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cl", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                              dstPtr,
                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                              newChannel,
                                                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
        }

        cl_int err = clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }

    return RPP_SUCCESS;
}

/******************** reconstruction_laplacian_image_pyramid ********************/

RppStatus
reconstruction_laplacian_image_pyramid_cl(cl_mem srcPtr1, RppiSize srcSize1, cl_mem srcPtr2, RppiSize srcSize2, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel, NULL, NULL);
    cl_mem gsout = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel, NULL, NULL);
    size_t gDim3[3];
    gDim3[0] = srcSize1.width;
    gDim3[1] = srcSize1.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* Resize the Source 2 */
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "resize.cl", "resize_pln", vld, vgd, "")(srcPtr2,
                                                                          gsin,
                                                                          srcSize2.height,
                                                                          srcSize2.width,
                                                                          srcSize1.height,
                                                                          srcSize1.width,
                                                                          channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "resize.cl", "resize_pln", vld, vgd, "")(srcPtr2,
                                                                          gsin,
                                                                          srcSize2.height,
                                                                          srcSize2.width,
                                                                          srcSize1.height,
                                                                          srcSize1.width,
                                                                          channel);
    }

    /* Gaussian Blur */
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);

    if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pkd", vld, vgd, "")(gsin,
                                                                                     gsout,
                                                                                     srcSize1.height,
                                                                                     srcSize1.width,
                                                                                     channel,
                                                                                     kernel,
                                                                                     kernelSize,
                                                                                     kernelSize);
    }
    else
    {
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                     gsout,
                                                                                     srcSize1.height,
                                                                                     srcSize1.width,
                                                                                     channel,
                                                                                     kernel,
                                                                                     kernelSize,
                                                                                     kernelSize);
    }

    /* Reconstruction of Laplacian Image pyramid */
    if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cl", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
                                                                                                                                          gsout,
                                                                                                                                          dstPtr,
                                                                                                                                          srcSize1.height,
                                                                                                                                          srcSize1.width,
                                                                                                                                          srcSize2.height,
                                                                                                                                          srcSize2.width,
                                                                                                                                          channel);
    }
    else
    {
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cl", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
                                                                                                                                          gsout,
                                                                                                                                          dstPtr,
                                                                                                                                          srcSize1.height,
                                                                                                                                          srcSize1.width,
                                                                                                                                          srcSize2.height,
                                                                                                                                          srcSize2.width,
                                                                                                                                          channel);
    }

    return RPP_SUCCESS;
}

RppStatus
reconstruction_laplacian_image_pyramid_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    unsigned int maxHeight1, maxWidth1, maxHeight2, maxWidth2, maxKernelSize;
    maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight1 < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth1 < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if (maxHeight2 < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if (maxWidth2 < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        if (maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    cl_mem srcPtr1Temp = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel, NULL, NULL);
    cl_mem srcPtr2Temp = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight2 * maxWidth2 * channel, NULL, NULL);
    cl_mem dstPtrTemp = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel, NULL, NULL);

    cl_mem gsin = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel, NULL, NULL);
    cl_mem gsout = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel, NULL, NULL);

    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, maxKernelSize * maxKernelSize * sizeof(Rpp32f), NULL, NULL);

    size_t gDim3[3];

    size_t batchIndex1 = 0, batchIndex2 = 0;

    for (int i = 0; i < handle.GetBatchSize(); i++)
    {

        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, srcPtr1Temp, batchIndex1, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr2, srcPtr2Temp, batchIndex2, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, 0, NULL, NULL);

        /* Resize the Source 2 */
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "resize.cl", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                              gsin,
                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                              channel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "resize.cl", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                              gsin,
                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                              channel);
        }

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pkd", vld, vgd, "")(gsin,
                                                                                         gsout,
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                         channel,
                                                                                         kernel,
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                         gsout,
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                         channel,
                                                                                         kernel,
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                         handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        }

        /* Reconstruction of Laplacian Image pyramid */
        if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cl", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1Temp,
                                                                                                                                              gsout,
                                                                                                                                              dstPtrTemp,
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                                                                              channel);
        }
        else
        {
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cl", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1Temp,
                                                                                                                                              gsout,
                                                                                                                                              dstPtrTemp,
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                                                                              channel);
        }

        cl_int err = clEnqueueCopyBuffer(handle.GetStream(), dstPtrTemp, dstPtr, 0, batchIndex1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);

        batchIndex1 += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndex2 += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
    }

    return RPP_SUCCESS;
}

/******************** tensor_convert_bit_depth ********************/

RppStatus
tensor_convert_bit_depth_cl(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, cl_mem srcPtr, cl_mem dstPtr, Rpp32u type, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
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
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }

    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    if (type == 1)
    {
        handle.AddKernel("", "", "tensor.cl", "tensor_convert_bit_depth_u8s8", vld, vgd, "")(tensorDimension,
                                                                                             srcPtr,
                                                                                             dstPtr,
                                                                                             dim1,
                                                                                             dim2,
                                                                                             dim3);
    }
    else if (type == 2)
    {
        handle.AddKernel("", "", "tensor.cl", "tensor_convert_bit_depth_u8u16", vld, vgd, "")(tensorDimension,
                                                                                              srcPtr,
                                                                                              dstPtr,
                                                                                              dim1,
                                                                                              dim2,
                                                                                              dim3);
    }
    else
    {
        handle.AddKernel("", "", "tensor.cl", "tensor_convert_bit_depth_u8s16", vld, vgd, "")(tensorDimension,
                                                                                              srcPtr,
                                                                                              dstPtr,
                                                                                              dim1,
                                                                                              dim2,
                                                                                              dim3);
    }
    return RPP_SUCCESS;
}

/******************** remap ********************/

RppStatus
remap_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    cl_mem rowRemapTableGPU = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp32u) * srcSize.height * srcSize.width, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), rowRemapTableGPU, CL_TRUE, 0, sizeof(Rpp32u) * srcSize.height * srcSize.width, rowRemapTable, 0, NULL, NULL);

    cl_mem colRemapTableGPU = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp32u) * srcSize.height * srcSize.width, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), colRemapTableGPU, CL_TRUE, 0, sizeof(Rpp32u) * srcSize.height * srcSize.width, colRemapTable, 0, NULL, NULL);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "remap.cl", "remap_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        rowRemapTableGPU,
                                                                        colRemapTableGPU,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "remap.cl", "remap_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        rowRemapTableGPU,
                                                                        colRemapTableGPU,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel);
    }

    return RPP_SUCCESS;
}

RppStatus
remap_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned long buffer_size = 0;

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for (int i = 0; i < nBatchSize; i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        buffer_size += handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * sizeof(Rpp32u);
    }

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem rowRemapTableGPU = clCreateBuffer(theContext, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    cl_mem colRemapTableGPU = clCreateBuffer(theContext, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), rowRemapTableGPU, CL_TRUE, 0, buffer_size, rowRemapTable, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), colRemapTableGPU, CL_TRUE, 0, buffer_size, colRemapTable, 0, NULL, NULL);
    cl_mem rowRemapTableGPU1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, maxWidth * maxHeight * sizeof(Rpp32u), NULL, NULL);
    cl_mem colRemapTableGPU1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, maxWidth * maxHeight * sizeof(Rpp32u), NULL, NULL);

    size_t gDim3[3];

    size_t batchIndex = 0;
    size_t mapIndex = 0;

    for (int i = 0; i < nBatchSize; i++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        clEnqueueCopyBuffer(handle.GetStream(), rowRemapTableGPU, rowRemapTableGPU1, mapIndex, 0, sizeof(unsigned int) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], 0, NULL, NULL);
        clEnqueueCopyBuffer(handle.GetStream(), colRemapTableGPU, colRemapTableGPU1, mapIndex, 0, sizeof(unsigned int) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], 0, NULL, NULL);

        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "remap.cl", "remap_pln", vld, vgd, "")(srcPtr1,
                                                                            dstPtr1,
                                                                            rowRemapTableGPU1,
                                                                            colRemapTableGPU1,
                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                            channel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "remap.cl", "remap_pkd", vld, vgd, "")(srcPtr1,
                                                                            dstPtr1,
                                                                            rowRemapTableGPU1,
                                                                            colRemapTableGPU1,
                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                            channel);
        }
        else
        {
            std::cerr << "Internal error: Unknown Channel format";
        }

        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        mapIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * sizeof(unsigned int);
    }
    clReleaseMemObject(srcPtr1);
    clReleaseMemObject(dstPtr1);
    clReleaseMemObject(rowRemapTableGPU1);
    clReleaseMemObject(colRemapTableGPU1);
    clReleaseMemObject(rowRemapTableGPU);
    clReleaseMemObject(colRemapTableGPU);
    return RPP_SUCCESS;
}

/******************** tensor_transpose ********************/

RppStatus
tensor_transpose_cl(cl_mem srcPtr, cl_mem dstPtr,  Rpp32u* in_dims, Rpp32u *perm, RPPTensorDataType data_type, rpp::Handle& handle)
{
    unsigned int out_dims[4];
    out_dims[0] = in_dims[perm[0]];
    out_dims[1] = in_dims[perm[1]];
    out_dims[2] = in_dims[perm[2]];
    out_dims[3] = in_dims[perm[3]];
    unsigned int in_strides[4], out_strides[4];
    in_strides[0] = in_dims[1] * in_dims[2] * in_dims[3];
    in_strides[1] = in_dims[2] * in_dims[3];
    in_strides[2] = in_dims[3];
    in_strides[3] = 1;
    out_strides[0] = out_dims[1] * out_dims[2] * out_dims[3];
    out_strides[1] = out_dims[2] * out_dims[3];
    out_strides[2] = out_dims[3];
    out_strides[3] = 1;
    cl_mem d_perm, d_out_strides, d_in_strides, d_out_dims;
    cl_context theContext;
    cl_int err;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    d_perm = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 4, NULL, NULL);
    d_in_strides = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 4, NULL, NULL);
    d_out_strides = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 4, NULL, NULL);
    d_out_dims = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 4, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), d_perm, CL_TRUE, 0, sizeof(unsigned int) * 4, perm, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), d_in_strides, CL_TRUE, 0, sizeof(unsigned int) * 4, in_strides, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), d_out_strides, CL_TRUE, 0, sizeof(unsigned int) * 4, out_strides, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), d_out_dims, CL_TRUE, 0, sizeof(unsigned int) * 4, out_dims, 0, NULL, NULL);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{out_dims[0], out_dims[1], out_dims[2] * out_dims[3]};
    std::string kernel_name = "tensor_transpose";
    if(data_type == RPPTensorDataType::U8)
        kernel_name = "tensor_transpose";
    if(data_type == RPPTensorDataType::FP32)
        kernel_name = "tensor_transpose_fp32";
    if(data_type == RPPTensorDataType::FP16)
        kernel_name = "tensor_transpose_fp16";
    if(data_type == RPPTensorDataType::I8)
        kernel_name = "tensor_transpose_int8";

    handle.AddKernel("", "", "tensor.cl", kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     d_out_dims,
                                                                     d_perm,
                                                                     d_out_strides,
                                                                     d_in_strides);

    return RPP_SUCCESS;
}

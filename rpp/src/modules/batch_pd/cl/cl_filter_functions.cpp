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

/******************** sobel_filter ********************/

RppStatus
sobel_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u sobelType, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cl", "sobel_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sobelType
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cl", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sobelType
                                                                        );
    }

    return RPP_SUCCESS;
}

RppStatus
sobel_filter_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "sobel.cl", "sobel_batch", vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/******************** box_filter ********************/

RppStatus
box_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float box_3x3[] = {
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    };
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(float)*3*3, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), filtPtr, CL_TRUE, 0, sizeof(float)*3*3, box_3x3, 0, NULL, NULL);
    kernelSize = 3;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convolution.cl", "naive_convolution_planar", vld, vgd, "")(srcPtr,
                                                                                             dstPtr,
                                                                                             filtPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel,
                                                                                             kernelSize);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convolution.cl", "naive_convolution_packed", vld, vgd, "")(srcPtr,
                                                                                             dstPtr,
                                                                                             filtPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel,
                                                                                             kernelSize);
    }

    clReleaseMemObject(filtPtr);

    return RPP_SUCCESS;
}

RppStatus
box_filter_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "box_filter.cl", "box_filter_batch", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/******************** median_filter ********************/

RppStatus
median_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "median_filter.cl", "median_filter_pkd", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        channel,
                                                                                        kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "median_filter.cl", "median_filter_pln", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        channel,
                                                                                        kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
median_filter_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "median_filter.cl", "median_filter_batch", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/******************** non_max_suppression ********************/

RppStatus
non_max_suppression_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "non_max_suppression.cl", "non_max_suppression_pkd", vld, vgd, "")(srcPtr,
                                                                                                    dstPtr,
                                                                                                    srcSize.height,
                                                                                                    srcSize.width,
                                                                                                    channel,
                                                                                                    kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "non_max_suppression.cl", "non_max_suppression_pln", vld, vgd, "")(srcPtr,
                                                                                                    dstPtr,
                                                                                                    srcSize.height,
                                                                                                    srcSize.width,
                                                                                                    channel,
                                                                                                    kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
non_max_suppression_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "non_max_suppression.cl", "non_max_suppression_batch", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/******************** bilateral_filter ********************/

RppStatus
bilateral_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, unsigned int filterSize, double sigmaI, double sigmaS, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_int err;
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "bilateral_filter.cl", "bilateral_filter_planar", vld, vgd, "")(srcPtr,
                                                                                                 dstPtr,
                                                                                                 srcSize.height,
                                                                                                 srcSize.width,
                                                                                                 channel,
                                                                                                 sigmaI,
                                                                                                 sigmaS);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "bilateral_filter.cl", "bilateral_filter_packed", vld, vgd, "")(srcPtr,
                                                                                                 dstPtr,
                                                                                                 srcSize.height,
                                                                                                 srcSize.width,
                                                                                                 channel,
                                                                                                 sigmaI,
                                                                                                 sigmaS);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
bilateral_filter_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "bilateral_filter.cl", "bilateral_filter_batch", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                                            handle.GetInitHandle()->mem.mgpu.doubleArr[1].doublemem,
                                                                                            handle.GetInitHandle()->mem.mgpu.doubleArr[2].doublemem,
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

/******************** gaussian_filter ********************/

RppStatus
gaussian_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_pln", vld, vgd, "")(srcPtr,
                                                                                     dstPtr,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     channel,
                                                                                     kernel,
                                                                                     kernelSize,
                                                                                     kernelSize);
    }

    free(kernelMain);
    clReleaseMemObject(kernel);

    return RPP_SUCCESS;
}

RppStatus
gaussian_filter_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "gaussian_filter.cl", "gaussian_filter_batch", vld, vgd, "")(srcPtr,
                                                                                          dstPtr,
                                                                                          handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                                          handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/******************** custom_convolution ********************/

RppStatus
custom_convolution_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f* kernel, RppiSize kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem clkernel = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(Rpp32f)*kernelSize.height*kernelSize.width, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), clkernel, CL_TRUE, 0, sizeof(Rpp32f)*kernelSize.height*kernelSize.width, kernel, 0, NULL, NULL);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "custom_convolution.cl", "custom_convolution_pkd", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel,
                                                                                                  clkernel,
                                                                                                  kernelSize.height,
                                                                                                  kernelSize.width);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "custom_convolution.cl", "custom_convolution_pln", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel,
                                                                                                  clkernel,
                                                                                                  kernelSize.height,
                                                                                                  kernelSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
custom_convolution_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp32f *kernel, RppiSize KernelSize, rpp::Handle& handle,RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context ctx;
    Rpp32u nbatchSize = handle.GetBatchSize();
    int buffer_size_kernel_size = nbatchSize * sizeof(float) * KernelSize.height * KernelSize.width;
    int plnpkdind;
    cl_mem d_kernel;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);
    d_kernel = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size_kernel_size, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), d_kernel, CL_FALSE, 0, buffer_size_kernel_size, kernel, 0, NULL, NULL);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "custom_convolution.cl", "custom_convolution_batch", vld, vgd, "")(srcPtr,
                                                                                                dstPtr,
                                                                                                d_kernel,
                                                                                                KernelSize.height,
                                                                                                KernelSize.width,
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
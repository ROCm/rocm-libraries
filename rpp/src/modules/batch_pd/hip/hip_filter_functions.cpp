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

/******************** sobel_filter ********************/

RppStatus
sobel_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u sobelType, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cpp", "sobel_pkd", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         channel,
                                                                         sobelType);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         channel,
                                                                         sobelType);
    }

    return RPP_SUCCESS;
}

RppStatus
sobel_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_sobel_filter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** box_filter ********************/

RppStatus
box_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float box_3x3[] = {
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    };
    float *filtPtr;
    hipMalloc(&filtPtr, sizeof(float) * 3 * 3);
    hipMemcpy(filtPtr, box_3x3, sizeof(float) * 3 * 3, hipMemcpyHostToDevice);
    kernelSize = 3;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convolution.cpp", "naive_convolution_planar", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "convolution.cpp", "naive_convolution_packed", vld, vgd, "")(srcPtr,
                                                                                              dstPtr,
                                                                                              filtPtr,
                                                                                              srcSize.height,
                                                                                              srcSize.width,
                                                                                              channel,
                                                                                              kernelSize);
    }

    hipFree(filtPtr);

    return RPP_SUCCESS;
}

RppStatus
box_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_box_filter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** median_filter ********************/

RppStatus
median_filter_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

        handle.AddKernel("", "", "median_filter.cpp", "median_filter_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "median_filter.cpp", "median_filter_pln", vld, vgd, "")(srcPtr,
                                                                                         dstPtr,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel,
                                                                                         kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
median_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_median_filter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** non_max_suppression ********************/

RppStatus
non_max_suppression_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

        handle.AddKernel("", "", "non_max_suppression.cpp", "non_max_suppression_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "non_max_suppression.cpp", "non_max_suppression_pln", vld, vgd, "")(srcPtr,
                                                                                                     dstPtr,
                                                                                                     srcSize.height,
                                                                                                     srcSize.width,
                                                                                                     channel,
                                                                                                     kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
non_max_suppression_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_non_max_suppression_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** bilateral_filter ********************/

RppStatus
bilateral_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, unsigned int filterSize, double sigmaI, double sigmaS, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "bilateral_filter.cpp", "bilateral_filter_planar", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "bilateral_filter.cpp", "bilateral_filter_packed", vld, vgd, "")(srcPtr,
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
bilateral_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_bilateral_filter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** gaussian_filter ********************/

RppStatus
gaussian_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    Rpp32f *kernel;
    hipMemcpy(kernel, kernelMain, sizeof(Rpp32f) * kernelSize * kernelSize, hipMemcpyHostToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      srcSize.height,
                                                                                      srcSize.width,
                                                                                      channel,
                                                                                      kernel,
                                                                                      kernelSize,
                                                                                      kernelSize);
    }

    free(kernelMain);
    hipFree(kernel);

    return RPP_SUCCESS;
}

RppStatus
gaussian_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_gaussian_filter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** custom_convolution ********************/

RppStatus
custom_convolution_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32f *kernel, RppiSize kernelSize, rpp::Handle& handle,RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int buffer_size_kernel_size = nbatchSize * sizeof(float) * kernelSize.height * kernelSize.width;
    Rpp32f *d_kernel;
    hipMalloc(&d_kernel, buffer_size_kernel_size);
    hipMemcpy(d_kernel, kernel, buffer_size_kernel_size, hipMemcpyHostToDevice);
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_custom_convolution_batch(srcPtr, dstPtr, handle, d_kernel, kernelSize, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
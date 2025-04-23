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

/******************** data_object_copy ********************/

RppStatus
data_object_copy_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);

    return RPP_SUCCESS;
}

RppStatus
data_object_copy_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned long buffer_size = 0;
    for(int i =0; i< handle.GetBatchSize(); i++)
    {
        buffer_size += handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] * channel;
    }
    hipMemcpy(dstPtr, srcPtr, buffer_size * sizeof(unsigned char), hipMemcpyDeviceToDevice);

    return RPP_SUCCESS;
}

/******************** local_binary_pattern ********************/

RppStatus
local_binary_pattern_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "local_binary_pattern.cpp", "local_binary_pattern_pkd", vld, vgd, "")(srcPtr,
                                                                                                       dstPtr,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "local_binary_pattern.cpp", "local_binary_pattern_pln", vld, vgd, "")(srcPtr,
                                                                                                       dstPtr,
                                                                                                       srcSize.height,
                                                                                                       srcSize.width,
                                                                                                       channel);
    }

    return RPP_SUCCESS;
}

RppStatus
local_binary_pattern_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_local_binary_pattern_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** gaussian_image_pyramid ********************/

RppStatus
gaussian_image_pyramid_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel,kernelMain,kernelSize * kernelSize * sizeof(Rpp32f),hipMemcpyHostToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
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
gaussian_image_pyramid_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_gaussian_image_pyramid_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** control_flow ********************/

RppStatus
control_flow_hip(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    switch(type)
    {
        case 1:
            handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                                     srcPtr2,
                                                                                     dstPtr,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     channel);
            break;
        case 2:
            handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                                       srcPtr2,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
            break;
        case 3:
            handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                                       srcPtr2,
                                                                                       dstPtr,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);
            break;
        case 4:
            handle.AddKernel("", "", "add.cpp", "add", vld, vgd, "")(srcPtr1,
                                                                     srcPtr2,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel);
            break;
        case 5:
            handle.AddKernel("", "", "subtract.cpp", "subtract", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);
            break;
        case 6:
            handle.AddKernel("", "", "multiply.cpp", "multiply", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);
            break;
        case 7:
            handle.AddKernel("", "", "min.cpp", "min", vld, vgd, "")(srcPtr1,
                                                                     srcPtr2,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel);
            break;
        case 8:
            handle.AddKernel("", "", "max.cpp", "max", vld, vgd, "")(srcPtr1,
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
control_flow_hip_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, Rpp32u type, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    switch(type)
    {
        case 1:
            hip_exec_bitwise_AND_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 2:
            hip_exec_inclusive_OR_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 3:
            hip_exec_exclusive_OR_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 4:
            hip_exec_add_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 5:
            hip_exec_subtract_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 6:
            hip_exec_multiply_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 7:
            hip_exec_min_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        case 8:
            hip_exec_max_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);
            break;
        default:
            break;
    }

    return RPP_SUCCESS;
}

/******************** laplacian_image_pyramid ********************/

RppStatus
laplacian_image_pyramid_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel,kernelMain,kernelSize * kernelSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1,  srcSize.height * srcSize.width * channel * sizeof(Rpp8u));

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
                                                                                                           srcPtr1,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           channel,
                                                                                                           kernel,
                                                                                                           kernelSize,
                                                                                                           kernelSize);
    }

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
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
        handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
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
laplacian_image_pyramid_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));
    Rpp8u *srcPtr1;
    hipMalloc(&srcPtr1, max_height * max_width * channel * sizeof(Rpp8u));
    Rpp32f *kernel;
    hipMalloc(&kernel, maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    Rpp32u batchIndex = 0;
    for(int i = 0 ; i < handle.GetBatchSize(); i++)
    {
        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        hipMemcpy(kernel,kernelMain,handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f), hipMemcpyHostToDevice);

        if(chnFormat == RPPI_CHN_PACKED)
        {
            hip_exec_gaussian_image_pyramid_pkd_batch(srcPtr, srcPtr1, handle, chnFormat, channel, kernel, max_height, max_width, batchIndex, i);
        }
        else
        {
            hip_exec_gaussian_image_pyramid_pln_batch(srcPtr, srcPtr1, handle, chnFormat, channel, kernel, max_height, max_width, batchIndex, i);
        }

        if(chnFormat == RPPI_CHN_PACKED)
        {
            hip_exec_laplacian_image_pyramid_pkd_batch(srcPtr1, dstPtr, handle, chnFormat, channel, kernel, max_height, max_width, batchIndex, i);
        }
        else
        {
            hip_exec_laplacian_image_pyramid_pln_batch(srcPtr1, dstPtr, handle, chnFormat, channel, kernel, max_height, max_width, batchIndex, i);
        }

        batchIndex += max_height * max_width * channel;
    }

    hipFree(srcPtr1);
    hipFree(kernel);

    return RPP_SUCCESS;
}

/******************** canny_edge_detector ********************/

RppStatus
canny_edge_detector_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp8u minThreshold, Rpp8u maxThreshold, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * srcSize.height * srcSize.width);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                  gsin,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
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
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                         tempDest1,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelType);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                         tempDest1,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelType);
    }
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                         sobelX,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelTypeX);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                         sobelX,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelTypeX);
    }
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                         sobelY,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelTypeY);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                         sobelY,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         newChannel,
                                                                         sobelTypeY);
    }

    handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_non_max_suppression", vld, vgd, "")(tempDest1,
                                                                                                 sobelX,
                                                                                                 sobelY,
                                                                                                 tempDest2,
                                                                                                 srcSize.height,
                                                                                                 srcSize.width,
                                                                                                 newChannel,
                                                                                                 minThreshold,
                                                                                                 maxThreshold);
    if(channel == 1)
    {
        handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                        dstPtr,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        newChannel,
                                                                                        minThreshold,
                                                                                        maxThreshold);
    }
    else
    {
        handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                        gsout,
                                                                                        srcSize.height,
                                                                                        srcSize.width,
                                                                                        newChannel,
                                                                                        minThreshold,
                                                                                        maxThreshold);
    }
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln1_to_pkd3", vld, vgd, "")(gsout,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln1_to_pln3", vld, vgd, "")(gsout,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
        }
    }

    hipFree(gsin);
    hipFree(gsout);
    hipFree(tempDest1);
    hipFree(tempDest2);
    hipFree(sobelX);
    hipFree(sobelY);

    return RPP_SUCCESS;
}

RppStatus
canny_edge_detector_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp32u imageDim = maxHeight * maxWidth;

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * imageDim);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * imageDim);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * imageDim);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * imageDim);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * imageDim);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * imageDim);

    unsigned long batchIndex = 0;
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * imageDim * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * imageDim * channel);

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        hipMemcpy(srcPtr1, srcPtr + batchIndex, sizeof(unsigned char) * imageDim * channel, hipMemcpyHostToDevice);
        size_t gDim3[3];
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;
        unsigned int newChannel = 1;

        if(channel == 1)
        {
            hip_exec_sobel_pln(srcPtr1, tempDest1, maxHeight, maxWidth, handle, newChannel, sobelType);
            hip_exec_sobel_pln(srcPtr1, sobelX, maxHeight, maxWidth, handle, newChannel, sobelTypeX);
            hip_exec_sobel_pln(srcPtr1, sobelY, maxHeight, maxWidth, handle, newChannel, sobelTypeY);
            hip_exec_ced_non_max_suppression(tempDest1, sobelX, sobelY, tempDest2, maxHeight, maxWidth, handle, newChannel, i);
            hip_exec_canny_edge(tempDest2, dstPtr1, maxHeight, maxWidth, handle, newChannel, i);
        }
        else if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                hip_exec_canny_ced_pkd3_to_pln1(srcPtr1, gsin, maxHeight, maxWidth, handle, channel);
            }
            else
            {
                hip_exec_canny_ced_pln3_to_pln1(srcPtr1, gsin, maxHeight, maxWidth, handle, channel);
            }
            hip_exec_sobel_pln(gsin, tempDest1, maxHeight, maxWidth, handle, newChannel, sobelType);
            hip_exec_sobel_pln(gsin, sobelX, maxHeight, maxWidth, handle, newChannel, sobelTypeX);
            hip_exec_sobel_pln(gsin, sobelY, maxHeight, maxWidth, handle, newChannel, sobelTypeY);
            hip_exec_ced_non_max_suppression(tempDest1, sobelX, sobelY, tempDest2, maxHeight, maxWidth, handle, newChannel, i);
            hip_exec_canny_edge(tempDest2, gsout, maxHeight, maxWidth, handle, newChannel, i);
            if(chnFormat == RPPI_CHN_PACKED)
            {
                hip_exec_canny_ced_pln1_to_pkd3(gsout, dstPtr1, maxHeight, maxWidth, handle, channel);
            }
            else
            {
                hip_exec_canny_ced_pln1_to_pln3(gsout, dstPtr1, maxHeight, maxWidth, handle, channel);
            }
        }
        hipMemcpy(dstPtr + batchIndex, dstPtr1, sizeof(unsigned char) * imageDim * channel, hipMemcpyDeviceToHost);
        batchIndex += imageDim * channel;
    }

    return RPP_SUCCESS;
}

/******************** harris_corner_detector ********************/

RppStatus
harris_corner_detector_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr,
                            Rpp32u gaussianKernelSize, Rpp32f stdDev, Rpp32u kernelSize, Rpp32f kValue,
                            Rpp32f threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    /* SETTING UP */
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp32f* dstFloat;
    hipMalloc(&dstFloat, sizeof(float) * srcSize.height * srcSize.width);
    Rpp32f* nonMaxDstFloat;
    hipMalloc(&nonMaxDstFloat, sizeof(float) * srcSize.height * srcSize.width);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * srcSize.height * srcSize.width);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* RGB to GREY SCALE */

    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                   gsin,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   channel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
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
    Rpp32f* kernel;
    hipMalloc(&kernel,  gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel, kernelMain,gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f),hipMemcpyHostToDevice);

    if(channel == 1)
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
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

    handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                     sobelX,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     newChannel,
                                                                     sobelTypeX);

    handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                     sobelY,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     newChannel,
                                                                     sobelTypeY);

    /* HARRIS CORNER STRENGTH MATRIX */

    handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_strength", vld, vgd, "")(sobelX,
                                                                                                            sobelY,
                                                                                                            dstFloat,
                                                                                                            srcSize.height,
                                                                                                            srcSize.width,
                                                                                                            newChannel,
                                                                                                            kernelSize,
                                                                                                            kValue,
                                                                                                            threshold);

    /* NON-MAX SUPRESSION */

    handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_nonmax_supression", vld, vgd, "")(dstFloat,
                                                                                                                     nonMaxDstFloat,
                                                                                                                     srcSize.height,
                                                                                                                     srcSize.width,
                                                                                                                     newChannel,
                                                                                                                     nonmaxKernelSize);

    hipMemcpy(dstPtr, srcPtr, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                           nonMaxDstFloat,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           channel);
    }
    else
    {
        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                           nonMaxDstFloat,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           channel);
    }

    return RPP_SUCCESS;
}

RppStatus
harris_corner_detector_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    /* SETTING UP */

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

    Rpp32f* kernel;
    hipMalloc(&kernel,  maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp32f* dstFloat;
    hipMalloc(&dstFloat, sizeof(float) * maxHeight * maxWidth);
    Rpp32f* nonMaxDstFloat;
    hipMalloc(&nonMaxDstFloat, sizeof(float) * maxHeight * maxWidth);

    hipMemcpy(dstPtr, srcPtr, sizeof(unsigned char) * ioBufferSize, hipMemcpyDeviceToDevice);

    unsigned long batchIndex = 0;
    Rpp8u *srcPtr1, *dstPtr1;
    hipMalloc(&srcPtr1, sizeof(Rpp8u) * singleImageSize);
    hipMalloc(&dstPtr1, sizeof(Rpp8u) * singleImageSize);

    size_t gDim3[3];

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        hipMemcpy(srcPtr1, srcPtr + batchIndex, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                hip_exec_canny_ced_pkd3_to_pln1(srcPtr1, gsin, maxHeight, maxWidth, handle, channel);
            }
            else
            {
                hip_exec_canny_ced_pln3_to_pln1(srcPtr1, gsin, maxHeight, maxWidth, handle, channel);
            }
        }

        unsigned int newChannel = 1;

        /* GAUSSIAN FILTER */

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        hipMemcpy(kernel, kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * sizeof(Rpp32f),hipMemcpyHostToDevice);

        if(channel == 1)
        {
            hip_exec_gaussian_pln(srcPtr1, tempDest1, maxHeight, maxWidth, kernel, handle, newChannel, i);
        }
        else
        {
            hip_exec_gaussian_pln(gsin, tempDest1, maxHeight, maxWidth, kernel, handle, newChannel, i);
        }

        /* SOBEL X and Y */

        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;

        hip_exec_sobel_pln(tempDest1, sobelX, maxHeight, maxWidth, handle, newChannel, sobelTypeX);
        hip_exec_sobel_pln(tempDest1, sobelY, maxHeight, maxWidth, handle, newChannel, sobelTypeY);

        /* HARRIS CORNER STRENGTH MATRIX */

        hip_exec_harris_corner_detector_strength(sobelX, sobelY, dstFloat, maxHeight, maxWidth, handle, newChannel, i);

        /* NON-MAX SUPRESSION */

        hip_exec_harris_corner_detector_nonmax_supression(dstFloat, nonMaxDstFloat, maxHeight, maxWidth, handle, newChannel, i);

        hipMemcpy(dstPtr1, srcPtr1, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);

        if(chnFormat == RPPI_CHN_PACKED)
        {
            hip_exec_harris_corner_detector_pkd(dstPtr1, nonMaxDstFloat, maxHeight, maxWidth, handle, channel);
        }
        else
        {
            hip_exec_harris_corner_detector_pln(dstPtr1, nonMaxDstFloat, maxHeight, maxWidth, handle, channel);
        }

        hipMemcpy(dstPtr + batchIndex, dstPtr1, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);
        batchIndex += maxHeight * maxWidth * channel;
    }

    return RPP_SUCCESS;
}

/******************** tensor_transpose ********************/

RppStatus
tensor_transpose_hip_u8(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u *in_dims, Rpp32u *perm, rpp::Handle& handle)
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

    Rpp32u *d_perm, *d_in_strides, *d_out_strides, *d_out_dims;
    hipMalloc(&d_perm, 4 * sizeof(Rpp32u));
    hipMalloc(&d_in_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_dims, 4 * sizeof(Rpp32u));
    hipMemcpy(d_perm, perm, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_dims, out_dims, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);

    hip_exec_tensor_transpose(srcPtr, dstPtr, d_out_dims, d_perm, d_out_strides, d_in_strides, out_dims, handle);

    return RPP_SUCCESS;
}

RppStatus
tensor_transpose_hip_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, Rpp32u *in_dims, Rpp32u *perm, rpp::Handle& handle)
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

    Rpp32u *d_perm, *d_in_strides, *d_out_strides, *d_out_dims;
    hipMalloc(&d_perm, 4 * sizeof(Rpp32u));
    hipMalloc(&d_in_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_dims, 4 * sizeof(Rpp32u));
    hipMemcpy(d_perm, perm, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_dims, out_dims, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);

    hip_exec_tensor_transpose_fp16(srcPtr, dstPtr, d_out_dims, d_perm, d_out_strides, d_in_strides, out_dims, handle);

    return RPP_SUCCESS;
}

RppStatus
tensor_transpose_hip_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, Rpp32u *in_dims, Rpp32u *perm, rpp::Handle& handle)
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

    Rpp32u *d_perm, *d_in_strides, *d_out_strides, *d_out_dims;
    hipMalloc(&d_perm, 4 * sizeof(Rpp32u));
    hipMalloc(&d_in_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_dims, 4 * sizeof(Rpp32u));
    hipMemcpy(d_perm, perm, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_dims, out_dims, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);

    hip_exec_tensor_transpose_fp32(srcPtr, dstPtr, d_out_dims, d_perm, d_out_strides, d_in_strides, out_dims, handle);

    return RPP_SUCCESS;
}

RppStatus
tensor_transpose_hip_i8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp32u *in_dims, Rpp32u *perm, rpp::Handle& handle)
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

    Rpp32u *d_perm, *d_in_strides, *d_out_strides, *d_out_dims;
    hipMalloc(&d_perm, 4 * sizeof(Rpp32u));
    hipMalloc(&d_in_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_dims, 4 * sizeof(Rpp32u));
    hipMemcpy(d_perm, perm, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_dims, out_dims, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);

    hip_exec_tensor_transpose_int8(srcPtr, dstPtr, d_out_dims, d_perm, d_out_strides, d_in_strides, out_dims, handle);

    return RPP_SUCCESS;
}

/******************** fast_corner_detector ********************/

RppStatus
fast_corner_detector_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32u numOfPixels, Rpp8u threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* RGB to GS */
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                   gsin,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   channel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                   gsin,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   channel);
        }
    }

    /* FAST CORNER IMPLEMENTATION */

    unsigned int newChannel = 1;

    if(channel == 1)
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                   tempDest1,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   newChannel,
                                                                                                   threshold,
                                                                                                   numOfPixels);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                  tempDest1,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  newChannel,
                                                                                                  threshold,
                                                                                                  numOfPixels);
    }

    /* NON MAX SUPRESSION */

    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                           dstPtr,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           newChannel,
                                                                                                           nonmaxKernelSize);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                           dstPtr,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           newChannel,
                                                                                                           nonmaxKernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
fast_corner_detector_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * maxHeight * maxWidth);

    size_t gDim3[3];
    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        hipMemcpy( srcPtr1, srcPtr+batchIndex,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        /* RGB to GS */

        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                       gsin,
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                       channel);
            }
            else
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                       gsin,
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                       channel);
            }
        }

        /* FAST CORNER IMPLEMENTATION */

        unsigned int newChannel = 1;

        if(channel == 1)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                       tempDest1,
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                       newChannel,
                                                                                                       handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                       handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                       tempDest1,
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                       newChannel,
                                                                                                       handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                       handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }

        /* NON MAX SUPRESSION */

        hipMemcpy(dstPtr1, srcPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                               dstPtr,
                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                               newChannel,
                                                                                                               handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                               dstPtr,
                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                               newChannel,
                                                                                                               handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
        }

        hipMemcpy( dstPtr+batchIndex, dstPtr1,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }

    return RPP_SUCCESS;
}

/******************** reconstruction_laplacian_image_pyramid ********************/

RppStatus
reconstruction_laplacian_image_pyramid_hip(Rpp8u* srcPtr1, RppiSize srcSize1, Rpp8u* srcPtr2, RppiSize srcSize2, Rpp8u* dstPtr,
                                            Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel);
    size_t gDim3[3];
    gDim3[0] = srcSize1.width;
    gDim3[1] = srcSize1.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    /* Resize source 2 */

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2,
                                                                           gsin,
                                                                           srcSize2.height,
                                                                           srcSize2.width,
                                                                           srcSize1.height,
                                                                           srcSize1.width,
                                                                           channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2,
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
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel, kernelMain,kernelSize * kernelSize * sizeof(Rpp32f), hipMemcpyHostToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(gsin,
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
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                      gsout,
                                                                                      srcSize1.height,
                                                                                      srcSize1.width,
                                                                                      channel,
                                                                                      kernel,
                                                                                      kernelSize,
                                                                                      kernelSize);
    }

    /* Reconstruction of Laplacian Image pyramid */

    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
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
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
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
reconstruction_laplacian_image_pyramid_hip_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned int maxHeight1, maxWidth1, maxHeight2, maxWidth2, maxKernelSize;
    maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight1 < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth1 < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if(maxHeight2 < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(maxWidth2 < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        if(maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    Rpp8u* srcPtr1Temp;
    hipMalloc(&srcPtr1Temp, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);
    Rpp8u* srcPtr2Temp;
    hipMalloc(&srcPtr2Temp, sizeof(unsigned char) * maxHeight2 * maxWidth2 * channel);
    Rpp8u* dstPtrTemp;
    hipMalloc(&dstPtrTemp, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);

    Rpp32f* kernel;
    hipMalloc(&kernel,  maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    size_t gDim3[3];

    size_t batchIndex1 = 0, batchIndex2 = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        hipMemcpy(srcPtr1Temp, srcPtr1+batchIndex1,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel,hipMemcpyDeviceToDevice);
        hipMemcpy( srcPtr2Temp, srcPtr2+batchIndex2,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, hipMemcpyDeviceToDevice);

        /* Resize the Source 2 */

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                               gsin,
                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                               channel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                               gsin,
                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                               channel);
        }

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        hipMemcpy(kernel, kernelMain,handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f),hipMemcpyHostToDevice);

        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(gsin,
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
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                                          gsout,
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                          channel,
                                                                                          kernel,
                                                                                          handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                          handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        }

        /* Reconstruction of Laplacian Image pyramid */

        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1Temp,
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
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1Temp,
                                                                                                                                               gsout,
                                                                                                                                               dstPtrTemp,
                                                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                                                                               handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                                                                               channel);
        }

        hipMemcpy( dstPtr+batchIndex1, dstPtrTemp,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        batchIndex1 += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndex2 += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
    }

    return RPP_SUCCESS;
}

/******************** convert_bit_depth ********************/

template <typename T, typename U>
RppStatus
convert_bit_depth_hip(T* srcPtr, RppiSize srcSize, U* dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(type == 1)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8s8", vld, vgd, "")(srcPtr,
                                                                                                  dstPtr,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  channel);
    }
    else if(type == 2)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8u16", vld, vgd, "")(srcPtr,
                                                                                                   dstPtr,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8s16", vld, vgd, "")(srcPtr,
                                                                                                   dstPtr,
                                                                                                   srcSize.height,
                                                                                                   srcSize.width,
                                                                                                   channel);
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus
convert_bit_depth_hip_batch(T* srcPtr, U* dstPtr, Rpp32u type,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    if (type == 1)
    {
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8s8", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8u16", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8s16", vld, vgd, "")(srcPtr,
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

/******************** tensor_convert_bit_depth ********************/

template <typename T, typename U>
RppStatus
tensor_convert_bit_depth_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, T* srcPtr, U* dstPtr, Rpp32u type, rpp::Handle& handle)
{
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
    if(type == 1)
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8s8", vld, vgd, "")(tensorDimension,
                                                                                              srcPtr,
                                                                                              dstPtr,
                                                                                              dim1,
                                                                                              dim2,
                                                                                              dim3);
    }
    else if(type == 2)
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8u16", vld, vgd, "")(tensorDimension,
                                                                                               srcPtr,
                                                                                               dstPtr,
                                                                                               dim1,
                                                                                               dim2,
                                                                                               dim3);
    }
    else
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8s16", vld, vgd, "")(tensorDimension,
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
remap_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
remap_hip_batch(Rpp8u *srcPtr, Rpp8u* dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
         rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    return RPP_SUCCESS;
}
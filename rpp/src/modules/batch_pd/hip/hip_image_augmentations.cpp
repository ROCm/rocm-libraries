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

/******************** brightness ********************/

RppStatus
brightness_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32f alpha, Rpp32s beta, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "brightness.cpp", "brightness", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           alpha,
                                                                           beta,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel);

    return RPP_SUCCESS;
}

RppStatus brightness_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_brightness_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus
gamma_correction_hip(Rpp8u *srcPtr,RppiSize srcSize, Rpp8u *dstPtr, float gamma, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "gamma_correction.cpp", "gamma_correction", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       gamma,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);

    return RPP_SUCCESS;
}

RppStatus
gamma_correction_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_gamma_correction_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** contrast ********************/

RppStatus
contrast_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32u newMin, Rpp32u newMax, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u min = 0;
    Rpp32u max = 255;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "contrast_stretch.cpp", "contrast_stretch", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       min,
                                                                                       max,
                                                                                       newMin,
                                                                                       newMax,
                                                                                       srcSize.height,
                                                                                       srcSize.width,
                                                                                       channel);

    return RPP_SUCCESS;
}

RppStatus
contrast_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    Rpp32u min = 0;
    Rpp32u max = 255;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_contrast_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width, min, max);

    return RPP_SUCCESS;
}

/******************** blend ********************/

RppStatus
blend_hip(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, float alpha, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "blend.cpp", "blend", vld, vgd, "")(srcPtr1,
                                                                 srcPtr2,
                                                                 dstPtr,
                                                                 srcSize.height,
                                                                 srcSize.width,
                                                                 alpha,
                                                                 channel);

    return RPP_SUCCESS;
}

RppStatus
blend_hip_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_blend_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** pixelate ********************/

RppStatus
pixelate_hip(Rpp8u *srcPtr, RppiSize srcSize,Rpp8u *dstPtr, RppiChnFormat chnFormat, unsigned int channel,rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
pixelate_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_pixelate_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** jitter ********************/

RppStatus
jitter_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, unsigned int kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "jitter.cpp", "jitter_pkd", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           kernelSize);
    }
    else if(chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "jitter.cpp", "jitter_pln", vld, vgd, "")(srcPtr,
                                                                           dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           kernelSize);
    }

    return RPP_SUCCESS;
}

RppStatus
jitter_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_jitter_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** noise ********************/

RppStatus
noise_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32f noiseProbability, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
noise_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_noise_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** snow ********************/

RppStatus
snow_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, float snowCoefficient, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(snowCoefficient == 0)
    {
        hipMemcpy(dstPtr, srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);
    }
    else
    {
        Rpp32u snowDrops= (Rpp32u)((snowCoefficient * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / snowDrops);
        size_t gDim3[3];
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = 1;

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cpp", "snow_pkd", vld, vgd, "")(dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           pixelDistance);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cpp", "snow_pln", vld, vgd, "")(dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           pixelDistance);
        }

        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "snow.cpp", "snow", vld, vgd, "")(srcPtr,
                                                                   dstPtr,
                                                                   srcSize.height,
                                                                   srcSize.width,
                                                                   channel);
    }

    return RPP_SUCCESS;
}

RppStatus
snow_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    hipMemcpy(dstPtr, srcPtr, nbatchSize * max_height * max_width * channel * sizeof(unsigned char), hipMemcpyDeviceToDevice);

    hip_exec_snow_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** exposure ********************/

RppStatus
exposure_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32f exposureValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,  channel};

    handle.AddKernel("", "", "exposure.cpp", "exposure", vld, vgd, "")(srcPtr,
                                                                       dstPtr,
                                                                       srcSize.height,
                                                                       srcSize.width,
                                                                       channel,
                                                                       exposureValue);

    return RPP_SUCCESS;
}

RppStatus
exposure_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_exposure_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** rain ********************/

RppStatus
rain_hip(Rpp8u *srcPtr, RppiSize srcSize,Rpp8u *dstPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(rainPercentage == 0)
    {
        hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel,hipMemcpyDeviceToDevice);
    }
    else
    {
        Rpp32u rainDrops= (Rpp32u)((rainPercentage * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / rainDrops);
        transparency /= 5;

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::cerr<<"\n Gonna call rain packed";
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cpp", "rain_pkd", vld, vgd, "")(dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           pixelDistance,
                                                                           rainWidth,
                                                                           rainHeight,
                                                                           transparency);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cpp", "rain_pln", vld, vgd, "")(dstPtr,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel,
                                                                           pixelDistance,
                                                                           rainWidth,
                                                                           rainHeight,
                                                                           transparency);
        }

        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height,channel};

        handle.AddKernel("", "", "rain.cpp", "rain", vld, vgd, "")(srcPtr,
                                                                   dstPtr,
                                                                   srcSize.height,
                                                                   srcSize.width,
                                                                   channel);
    }

    return RPP_SUCCESS;
}

RppStatus
rain_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    hipMemcpy(dstPtr, srcPtr, max_height * max_width * channel * nbatchSize, hipMemcpyDeviceToDevice);

    hip_exec_rain_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** fog ********************/

RppStatus
fog_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *temp, Rpp32f fogValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
fog_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_fog_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** random_shadow ********************/

RppStatus
random_shadow_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                    Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u row1, row2, column2, column1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "random_shadow.cpp","random_shadow", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);

    for(int i = 0 ; i < numberOfShadows ; i++)
    {
        do
        {
            row1 = rand() % srcSize.height;
            column1 = rand() % srcSize.width;
        } while (column1 <= x1 || column1 >= x2 || row1 <= y1 || row1 >= y2);
        do
        {
            row2 = rand() % srcSize.height;
            column2 = rand() % srcSize.width;
        } while (
            (row2 < row1 || column2 < column1) ||
            (column2 <= x1 || column2 >= x2 || row2 <= y1 || row2 >= y2) ||
            (row2 - row1 >= maxSizeY || column2 - column1 >= maxSizeX));

        if(RPPI_CHN_PACKED == chnFormat)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_packed", vld, vgd, "")(srcPtr,
                                                                                                dstPtr,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel,
                                                                                                column1,
                                                                                                row1,
                                                                                                column2,
                                                                                                row2);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_planar", vld, vgd, "")(srcPtr,
                                                                                                dstPtr,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel,
                                                                                                column1,
                                                                                                row1,
                                                                                                column2,
                                                                                                row2);
        }
    }

    return RPP_SUCCESS;
}

RppStatus
random_shadow_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned int maxHeight, maxWidth, maxKernelSize;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }
    Rpp8u *srcPtr1, *dstPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        Rpp32u row1, row2, column2, column1;
        int x, y;

        hipMemcpy(srcPtr1, srcPtr+batchIndex, sizeof(unsigned char) * maxWidth * maxHeight * channel, hipMemcpyDeviceToDevice);
        hipMemcpy(dstPtr1, srcPtr1,  sizeof(unsigned char) * maxWidth * maxHeight * channel, hipMemcpyDeviceToDevice);

        for(x = 0 ; x < handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i]; x++)
        {
            do
            {
                row1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while (
                column1 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] ||
                column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i] ||
                row1 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] ||
                row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]);

            do
            {
                row2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while (
                (row2 < row1 || column2 < column1) ||
                (column2 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] || column2 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i] ||
                row2 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] || row2 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]) ||
                (row2 - row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i] || column2 - column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]));

            if(RPPI_CHN_PACKED == chnFormat)
            {
                hip_exec_random_shadow_packed(srcPtr1, dstPtr1, handle, channel, column1, row1, column2, row2, i);
            }
            else
            {
                hip_exec_random_shadow_planar(srcPtr1, dstPtr1, handle, channel, column1, row1, column2, row2, i);
            }
        }
        hipMemcpy(dstPtr+batchIndex, dstPtr1, sizeof(unsigned char) * maxWidth * maxHeight * channel, hipMemcpyDeviceToDevice);
        batchIndex += maxHeight * maxWidth * channel * sizeof(unsigned char);
    }

    return RPP_SUCCESS;
}

/******************** histogram_balance ********************/

RppStatus
histogram_balance_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned int numGroups;

    size_t lDim3[3];
    size_t gDim3[3];
    int num_pixels_per_work_item = 16;

    gDim3[0] = srcSize.width / num_pixels_per_work_item ;// Plus 1
    gDim3[1] = srcSize.height / num_pixels_per_work_item ;
    lDim3[0] = num_pixels_per_work_item;
    lDim3[1] = num_pixels_per_work_item;
    gDim3[2] = 1;
    lDim3[2] = 1;

    numGroups = gDim3[0] * gDim3[1];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;

    Rpp8u* partialHistogram;
    hipMalloc(&partialHistogram,sizeof(unsigned int)*256*numGroups);
    Rpp8u* histogram;
    hipMalloc(&histogram,sizeof(unsigned int)*256);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                         partialHistogram,
                                                                                         srcSize.width,
                                                                                         srcSize.height,
                                                                                         channel);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                         partialHistogram,
                                                                                         srcSize.width,
                                                                                         srcSize.height,
                                                                                         channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    // // For sum histogram kernel
    gDim3[0] = 256;
    lDim3[0] = 256;
    gDim3[1] = 1;
    gDim3[2] = 1;
    lDim3[1] = 1;
    lDim3[2] = 1;
    std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

    handle.AddKernel("", "", "histogram.cpp", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                     histogram,
                                                                                     numGroups);

    Rpp8u* cum_histogram;
    hipMalloc(&cum_histogram,sizeof(unsigned int) * 256);

    // For scan kernel

    gDim3[0] = 256;
    gDim3[1] = 1;
    gDim3[2] = 1;
    lDim3[0] = 32;
    lDim3[1] = 1;
    lDim3[2] = 1;
    std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};

    handle.AddKernel("", "", "scan.cpp", "scan", vld1, vgd1, "")(histogram,
                                                                 cum_histogram);

    // For histogram equalize

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pln", vld, vgd, "")(srcPtr,
                                                                                          dstPtr,
                                                                                          cum_histogram,
                                                                                          srcSize.width,
                                                                                          srcSize.height,
                                                                                          channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pkd", vld, vgd, "")(srcPtr,
                                                                                          dstPtr,
                                                                                          cum_histogram,
                                                                                          srcSize.width,
                                                                                          srcSize.height,
                                                                                          channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    hipFree(cum_histogram);
    hipFree(partialHistogram);
    hipFree(histogram);

    return RPP_SUCCESS;
}

RppStatus
histogram_balance_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    int numGroups = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    Rpp8u* partialHistogram;
    hipMalloc(&partialHistogram,sizeof(unsigned int)*256*numGroups);
    Rpp8u* histogram;
    hipMalloc(&histogram,sizeof(unsigned int)*256);
    Rpp8u* cum_histogram;
    hipMalloc(&cum_histogram,sizeof(unsigned int)*256);
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1,sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1,sizeof(unsigned char)* maxHeight * maxWidth * channel);

    size_t gDim3[3];

    size_t batchIndex = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        size_t lDim3[3];
        size_t gDim3[3];
        int num_pixels_per_work_item = 16;

        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] / num_pixels_per_work_item ;
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] / num_pixels_per_work_item ;
        lDim3[0] = num_pixels_per_work_item;
        lDim3[1] = num_pixels_per_work_item;
        gDim3[2] = 1;
        lDim3[2] = 1;

        numGroups = gDim3[0] * gDim3[1];
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        hipMemcpy(srcPtr1, srcPtr+batchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                             partialHistogram,
                                                                                             handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                             handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                             channel);

        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                             partialHistogram,
                                                                                             handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                             handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                             channel);
        }
        else
        {
            std::cerr << "Internal error: Unknown Channel format";
        }

        // For sum histogram kernel

        gDim3[0] = 256;
        lDim3[0] = 256;
        gDim3[1] = 1;
        gDim3[2] = 1;
        lDim3[1] = 1;
        lDim3[2] = 1;
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "histogram.cpp", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                         histogram,
                                                                                         numGroups);

        // For scan kernel

        gDim3[0] = 256;
        gDim3[1] = 1;
        gDim3[2] = 1;
        lDim3[0] = 32;
        lDim3[1] = 1;
        lDim3[2] = 1;
        std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "scan.cpp", "scan", vld1, vgd1, "")(histogram,
                                                                     cum_histogram);

        // For histogram equalize

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pln", vld, vgd, "")(srcPtr1,
                                                                                              dstPtr1,
                                                                                              cum_histogram,
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                              channel);

        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pkd", vld, vgd, "")(srcPtr1,
                                                                                              dstPtr1,
                                                                                              cum_histogram,
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                              channel);
        }
        else
        {
            std::cerr << "Internal error: Unknown Channel format";
        }

        hipMemcpy(dstPtr+batchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }

    hipFree(cum_histogram);
    hipFree(partialHistogram);
    hipFree(histogram);

    return RPP_SUCCESS;
}
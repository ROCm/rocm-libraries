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

#ifndef HOST_COMPUTER_VISION_HPP
#define HOST_COMPUTER_VISION_HPP

#include "rpp_cpu_batchpd.hpp"

/**************** data_object_copy ***************/

template <typename T>
RppStatus data_object_copy_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                      Rpp32u nbatchSize,
                                      RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus data_object_copy_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_data_object_copy_host<Rpp8u>(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** local_binary_pattern ***************/

template <typename T>
RppStatus local_binary_pattern_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                          RppiROI *roiPoints, Rpp32u nbatchSize,
                                          RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32f kernelSize = 3;
            int bound = ((kernelSize - 1) / 2);
            RppiSize rppiKernelSize;
            rppiKernelSize.height = kernelSize;
            rppiKernelSize.width = kernelSize;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            RppiSize srcSizeBoundedROI;
            srcSizeBoundedROI.height = roiPoints[batchCount].roiHeight + (2 * bound);
            srcSizeBoundedROI.width = roiPoints[batchCount].roiWidth + (2 * bound);
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= bound) &&(y1 >= bound))
            {
                T *srcPtrImageTemp, *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;
                for (int c = 0; c < channel; c++)
                {
                    srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + (((Rpp32u) y1 - bound) * batch_srcSizeMax[batchCount].width) + ((Rpp32u) x1 - bound);
                    for (int i = 0; i < srcSizeBoundedROI.height; i++)
                    {
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, srcSizeBoundedROI.width * sizeof(T));
                        srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrBoundedROITemp += srcSizeBoundedROI.width;
                    }
                }
            }
            else
            {
                T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));

                T *srcPtrImageTemp, *srcPtrROITemp;
                srcPtrROITemp = srcPtrROI;
                for (int c = 0; c < channel; c++)
                {
                    srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                        srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrROITemp += srcSizeROI.width;
                    }
                }

                generate_evenly_padded_image_host(srcPtrROI, srcSizeROI, srcPtrBoundedROI, srcSizeBoundedROI, chnFormat, channel);

                free(srcPtrROI);
            }

            Rpp32u remainingElementsInRow = srcSizeBoundedROI.width - rppiKernelSize.width;
            Rpp32u centerPixelIncrement = batch_srcSize[batchCount].width + 1;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDim);
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32u roiRowCount = 0;


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrWindow, *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                                 remainingElementsInRow, srcPtrWindow + centerPixelIncrement,
                                                                 chnFormat, channel);

                                srcPtrWindow++;
                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtrTemp;

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                        roiRowCount++;
                    }
                }
            }

            free(srcPtrBoundedROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32f kernelSize = 3;
            int bound = ((kernelSize - 1) / 2);
            RppiSize rppiKernelSize;
            rppiKernelSize.height = kernelSize;
            rppiKernelSize.width = kernelSize;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            RppiSize srcSizeBoundedROI;
            srcSizeBoundedROI.height = roiPoints[batchCount].roiHeight + (2 * bound);
            srcSizeBoundedROI.width = roiPoints[batchCount].roiWidth + (2 * bound);
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowBoundedROI = channel * srcSizeBoundedROI.width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= bound) &&(y1 >= bound))
            {
                T *srcPtrImageTemp, *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;

                srcPtrImageTemp = srcPtrImage + (((Rpp32u) y1 - bound) * elementsInRowMax) + (channel * ((Rpp32u) x1 - bound));
                for (int i = 0; i < srcSizeBoundedROI.height; i++)
                {
                    memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowBoundedROI * sizeof(T));
                    srcPtrImageTemp += elementsInRowMax;
                    srcPtrBoundedROITemp += elementsInRowBoundedROI;
                }
            }
            else
            {
                T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));

                T *srcPtrImageTemp, *srcPtrROITemp;
                srcPtrROITemp = srcPtrROI;

                srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                    srcPtrImageTemp += elementsInRowMax;
                    srcPtrROITemp += elementsInRowROI;
                }

                generate_evenly_padded_image_host(srcPtrROI, srcSizeROI, srcPtrBoundedROI, srcSizeBoundedROI, chnFormat, channel);

                free(srcPtrROI);
            }

            Rpp32u remainingElementsInRow = (srcSizeBoundedROI.width - rppiKernelSize.width) * channel;
            Rpp32u centerPixelIncrement = channel * (batch_srcSize[batchCount].width + 1);
            Rpp32u roiRowCount = 0;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrWindow, *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    srcPtrWindow = srcPtrBoundedROI + (roiRowCount * elementsInRowBoundedROI);
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                            dstPtrTemp += channel;
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                                 remainingElementsInRow, srcPtrWindow + centerPixelIncrement,
                                                                 chnFormat, channel);

                                srcPtrWindow++;
                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                    roiRowCount++;
                }
            }

            free(srcPtrBoundedROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus local_binary_pattern_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u kernelSize = 3;
    Rpp32u bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u rowIncrementForWindow = kernelSize - 1;
        Rpp32u channelIncrementForWindow = (kernelSize - 1) * srcSizeMod.width;
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;
        Rpp32u centerPixelIncrement = srcSize.width + 1;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                    remainingElementsInRow, srcPtrWindow + centerPixelIncrement,
                                    chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (rowIncrementForWindow);
            }
            srcPtrWindow += (channelIncrementForWindow);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u rowIncrementForWindow = (kernelSize - 1) * channel;
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize) * channel;
        Rpp32u centerPixelIncrement = channel * (srcSize.width + 1);

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                    remainingElementsInRow, srcPtrWindow + centerPixelIncrement,
                                    chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += (rowIncrementForWindow);
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** convert_bit_depth ***************/

template <typename T, typename U>
RppStatus convert_bit_depth_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr,
                                       Rpp32u conversionType,
                                       Rpp32u nbatchSize,
                                       RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32u totalBufferSize = nbatchSize * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * channel;
    if (conversionType == 1)
    {
        Rpp32s val = 128;
        for (Rpp32u i = 0; i < totalBufferSize; i++)
        {
            *dstPtrTemp = (U) ((Rpp32s) *srcPtrTemp - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 2)
    {
        Rpp32f multiplier = 65535/255;
        for (Rpp32u i = 0; i < totalBufferSize; i++)
        {
            *dstPtrTemp = (U) ((Rpp32f) *srcPtrTemp * multiplier);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 3)
    {
        Rpp32f multiplier = 65535/255;
        Rpp32f val = 32768;
        for (Rpp32u i = 0; i < totalBufferSize; i++)
        {
            *dstPtrTemp = (U) (((Rpp32f) *srcPtrTemp * multiplier) - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus convert_bit_depth_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                 Rpp32u conversionType,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (conversionType == 1)
    {
        Rpp32s val = 128;
        for (int i = 0; i < srcSize.height * srcSize.width * channel; i++)
        {
            *dstPtrTemp = (U) ((Rpp32s) *srcPtrTemp - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 2)
    {
        Rpp32f multiplier = 65535/255;
        for (int i = 0; i < srcSize.height * srcSize.width * channel; i++)
        {
            *dstPtrTemp = (U) ((Rpp32f) *srcPtrTemp * multiplier);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 3)
    {
        Rpp32f multiplier = 65535/255;
        Rpp32f val = 32768;
        for (int i = 0; i < srcSize.height * srcSize.width * channel; i++)
        {
            *dstPtrTemp = (U) (((Rpp32f) *srcPtrTemp * multiplier) - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

/**************** remap ***************/

template <typename T>
RppStatus remap_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                Rpp32u *batch_rowRemapTable, Rpp32u *batch_colRemapTable,
                                Rpp32u nbatchSize,
                                RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;
            Rpp32u loc;

            T *srcPtrImage, *dstPtrImage;
            loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u *rowRemapTableImage, *colRemapTableImage;
            loc = 0;
            compute_image_location_host(batch_srcSize, batchCount, &loc, 1);
            rowRemapTableImage = batch_rowRemapTable + loc;
            colRemapTableImage = batch_colRemapTable + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32u *rowRemapTableChannel, *colRemapTableChannel;
                rowRemapTableChannel = rowRemapTableImage;
                colRemapTableChannel = colRemapTableImage;


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                    rowRemapTableTemp = rowRemapTableChannel + (i * batch_srcSize[batchCount].width);
                    colRemapTableTemp = colRemapTableChannel + (i * batch_srcSize[batchCount].width);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        *dstPtrTemp = *(srcPtrTemp + (*rowRemapTableTemp * batch_srcSizeMax[batchCount].width) + *colRemapTableTemp);

                        dstPtrTemp++;
                        rowRemapTableTemp++;
                        colRemapTableTemp++;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;
            Rpp32u loc;

            T *srcPtrImage, *dstPtrImage;
            loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u *rowRemapTableImage, *colRemapTableImage;
            loc = 0;
            compute_image_location_host(batch_srcSize, batchCount, &loc, 1);
            rowRemapTableImage = batch_rowRemapTable + loc;
            colRemapTableImage = batch_colRemapTable + loc;

            Rpp32u elementsInRemapTableRow = batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (i * elementsInRemapTableRow);
                colRemapTableTemp = colRemapTableImage + (i * elementsInRemapTableRow);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    srcPtrTemp = srcPtrImage;
                    for(int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *(srcPtrTemp + (*rowRemapTableTemp * elementsInRowMax) + (*colRemapTableTemp * channel));

                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus remap_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                     Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
                     RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    Rpp32u *rowRemapTableTemp, *colRemapTableTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        dstPtrTemp = dstPtr;
        for (int c = 0; c < channel; c++)
        {
            rowRemapTableTemp = rowRemapTable;
            colRemapTableTemp = colRemapTable;
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (*rowRemapTableTemp * srcSize.width) + *colRemapTableTemp);
                    dstPtrTemp++;
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        dstPtrTemp = dstPtr;
        rowRemapTableTemp = rowRemapTable;
        colRemapTableTemp = colRemapTable;
        Rpp32u elementsInRow = srcSize.width * channel;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                srcPtrTemp = srcPtr;
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (*rowRemapTableTemp * elementsInRow) + (*colRemapTableTemp * channel));
                    dstPtrTemp++;
                    srcPtrTemp++;
                }
                rowRemapTableTemp++;
                colRemapTableTemp++;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** gaussian_image_pyramid ***************/

template <typename T>
RppStatus gaussian_image_pyramid_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                            Rpp32f *batch_stdDev, Rpp32u *batch_kernelSize,
                                            Rpp32u nbatchSize,
                                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32f stdDev = batch_stdDev[batchCount];
        Rpp32u kernelSize = batch_kernelSize[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtrImage = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImage,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
        int bound = ((kernelSize - 1) / 2);

        generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

        RppiSize srcSizeMod;
        srcSizeMod.width = srcSize.width + (2 * bound);
        srcSizeMod.height = srcSize.height + (2 * bound);
        T *srcPtrImageMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

        generate_evenly_padded_image_host(srcPtrImage, srcSize, srcPtrImageMod, srcSizeMod, chnFormat, channel);

        RppiSize rppiKernelSize;
        rppiKernelSize.height = kernelSize;
        rppiKernelSize.width = kernelSize;
        T *srcPtrImageConvolved = (T *)calloc(channel * srcSize.height * srcSize.width, sizeof(T));
        convolve_image_host(srcPtrImageMod, srcSizeMod, srcPtrImageConvolved, srcSize, kernel, rppiKernelSize, chnFormat, channel);

        RppiSize dstSize;
        dstSize.height = (srcSize.height + 1) / 2;
        dstSize.width = (srcSize.width + 1) / 2;

        T *dstPtrImage = (T*) calloc(channel * dstSize.height * dstSize.width, sizeof(T));

        compute_downsampled_image_host(srcPtrImageConvolved, srcSize, dstPtrImage, dstSize, chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImage, dstSize, batch_srcSizeMax[batchCount], dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtrImage);
        free(kernel);
        free(srcPtrImageMod);
        free(srcPtrImageConvolved);
        free(dstPtrImage);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus gaussian_image_pyramid_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;
    T *srcPtrConvolved = (T *)calloc(channel * srcSize.height * srcSize.width, sizeof(T));
    convolve_image_host(srcPtrMod, srcSizeMod, srcPtrConvolved, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    RppiSize dstSize;
    dstSize.height = (srcSize.height + 1) / 2;
    dstSize.width = (srcSize.width + 1) / 2;

    compute_downsampled_image_host(srcPtrConvolved, srcSize, dstPtr, dstSize, chnFormat, channel);

    free(kernel);
    free(srcPtrMod);
    free(srcPtrConvolved);

    return RPP_SUCCESS;
}

/**************** canny_edge_detector ***************/

template <typename T>
RppStatus canny_edge_detector_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* batch_dstPtr,
                                         T *batch_maxThreshold, T *batch_minThreshold,
                                         Rpp32u nbatchSize,
                                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        T maxThreshold = batch_maxThreshold[batchCount];
        T minThreshold = batch_minThreshold[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        // RGB to Greyscale Conversion

        Rpp32u imageDim = srcSize.height * srcSize.width;

        T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
        T *srcPtrGreyscaleTemp;
        srcPtrGreyscaleTemp = srcPtrGreyscale;

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PLANAR)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + imageDim;
                srcPtrTempB = srcPtrTempG + imageDim;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
            }
            else if (chnFormat == RPPI_CHN_PACKED)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + 1;
                srcPtrTempB = srcPtrTempG + 1;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR += channel;
                    srcPtrTempG += channel;
                    srcPtrTempB += channel;
                }
            }
        }
        else if (channel == 1)
        {
            memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
        }

        Rpp32u newChannel = 1;

        RppiSize srcSizeMod, rppiKernelSize;
        Rpp32u kernelSize;
        int bound;

        // Sobel Filter

        kernelSize = 3;
        bound = (kernelSize - 1) / 2;

        srcSizeMod.width = srcSize.width + (2 * bound);
        srcSizeMod.height = srcSize.height + (2 * bound);

        T *dstPtrGreyscale = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));

        T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
        generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

        rppiKernelSize.height = kernelSize;
        rppiKernelSize.width = kernelSize;

        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        Rpp32s *dstPtrIntermediateX = (Rpp32s *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateX, srcSize, kernelX, rppiKernelSize, chnFormat, newChannel);

        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        Rpp32s *dstPtrIntermediateY = (Rpp32s *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateY, srcSize, kernelY, rppiKernelSize, chnFormat, newChannel);

        compute_magnitude_host(dstPtrIntermediateX, dstPtrIntermediateY, srcSize, dstPtrGreyscale, chnFormat, newChannel);

        // Find Image Maximum

        T *srcPtrTemp;
        srcPtrTemp = srcPtrGreyscale;
        Rpp8u max = *srcPtrTemp;
        for (int i = 0; i < (newChannel * srcSize.height * srcSize.width); i++)
        {
            if (*srcPtrTemp > max)
            {
                max = *srcPtrTemp;
            }
            srcPtrTemp++;
        }

        // Determine Gradients, Perform NMS, Double Thresholding and Edge Tracing by hysterisis

        Rpp32f gradient;
        Rpp32s *dstPtrIntermediateXTemp, *dstPtrIntermediateYTemp;
        dstPtrIntermediateXTemp = dstPtrIntermediateX;
        dstPtrIntermediateYTemp = dstPtrIntermediateY;

        T *srcPtrWindow, *dstPtrGreyscaleTemp, *srcPtrWindowCenter;
        srcPtrWindow = srcPtrMod;
        dstPtrGreyscaleTemp = dstPtrGreyscale;

        generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

        srcPtrWindowCenter = srcPtrWindow + (bound * srcSizeMod.width) + bound;
        Rpp32u toNeighborhood1 = 1;
        Rpp32u toNeighborhood2 = 2;
        Rpp32u toNeighborhood3 = srcSizeMod.width + 2;
        Rpp32u toNeighborhood4 = 2 * (srcSizeMod.width + 1);
        Rpp32u toNeighborhood5 = (2 * srcSizeMod.width) + 1;
        Rpp32u toNeighborhood6 = 2 * srcSizeMod.width;
        Rpp32u toNeighborhood7 = srcSizeMod.width;
        T *position1Ptr, *position2Ptr;
        dstPtrIntermediateXTemp = dstPtrIntermediateX;
        dstPtrIntermediateYTemp = dstPtrIntermediateY;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                gradient = atan((Rpp32f) *dstPtrIntermediateYTemp / (Rpp32f) *dstPtrIntermediateXTemp);
                if (gradient > 1.178097 || gradient < -1.178097)
                {
                    position1Ptr = srcPtrWindow + toNeighborhood1;
                    position2Ptr = srcPtrWindow + toNeighborhood5;
                }
                else if (gradient > 0.392699)
                {
                    position1Ptr = srcPtrWindow + toNeighborhood2;
                    position2Ptr = srcPtrWindow + toNeighborhood6;
                }
                else if (gradient < -0.392699)
                {
                    position1Ptr = srcPtrWindow;
                    position2Ptr = srcPtrWindow + toNeighborhood4;
                }
                else
                {
                    position1Ptr = srcPtrWindow + toNeighborhood3;
                    position2Ptr = srcPtrWindow + toNeighborhood7;
                }

                canny_non_max_suppression_kernel_host(dstPtrGreyscaleTemp, *srcPtrWindowCenter, position1Ptr, position2Ptr);

                if (*dstPtrGreyscaleTemp > maxThreshold)
                {
                    *dstPtrGreyscaleTemp = (T) 255;
                }
                else if (*dstPtrGreyscaleTemp < minThreshold)
                {
                    *dstPtrGreyscaleTemp = (T) 0;
                }
                else
                {
                    *dstPtrGreyscaleTemp = (T) 100;
                }

                srcPtrWindow++;
                srcPtrWindowCenter++;
                dstPtrGreyscaleTemp++;
                dstPtrIntermediateXTemp++;
                dstPtrIntermediateYTemp++;
            }
            srcPtrWindow += (kernelSize - 1);
            srcPtrWindowCenter += (kernelSize - 1);
        }

        srcPtrWindow = srcPtrMod;
        dstPtrGreyscaleTemp = dstPtrGreyscale;
        generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

        srcPtrWindowCenter = srcPtrWindow + (bound * srcSizeMod.width) + bound;
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                if (*srcPtrWindowCenter == (T) 100)
                {
                    canny_hysterisis_edge_tracing_kernel_host(srcPtrWindow, dstPtrGreyscaleTemp, srcSize,
                                    kernelSize, remainingElementsInRow, *srcPtrWindowCenter, bound,
                                    chnFormat, newChannel);
                }
                srcPtrWindow++;
                srcPtrWindowCenter++;
                dstPtrGreyscaleTemp++;
            }
            srcPtrWindow += (kernelSize - 1);
            srcPtrWindowCenter += (kernelSize - 1);
        }

        // Greyscale TO RGB Conversion

        dstPtrGreyscaleTemp = dstPtrGreyscale;
        T *dstPtrTemp;
        dstPtrTemp = dstPtr;

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PLANAR)
            {
                for (int c = 0; c < channel; c++)
                {
                    memcpy(dstPtrTemp, dstPtrGreyscaleTemp, imageDim * sizeof(T));
                    dstPtrTemp += imageDim;
                }
            }
            else if (chnFormat == RPPI_CHN_PACKED)
            {
                for (int i = 0; i < imageDim; i++)
                {
                    memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                    dstPtrTemp++;
                    memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                    dstPtrTemp++;
                    memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                    dstPtrTemp++;
                    dstPtrGreyscaleTemp++;
                }
            }
        }
        else if (channel == 1)
        {
            memcpy(dstPtr, dstPtrGreyscale, imageDim * sizeof(T));
        }

        compute_padded_from_unpadded_host(dstPtr, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], batch_dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtr);
        free(dstPtr);
        free(srcPtrGreyscale);
        free(dstPtrGreyscale);
        free(srcPtrMod);
        free(kernelX);
        free(dstPtrIntermediateX);
        free(kernelY);
        free(dstPtrIntermediateY);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus canny_edge_detector_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                   T maxThreshold, T minThreshold,
                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    // RGB to Greyscale Conversion

    Rpp32u imageDim = srcSize.height * srcSize.width;

    T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
    T *srcPtrGreyscaleTemp;
    srcPtrGreyscaleTemp = srcPtrGreyscale;

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + imageDim;
            srcPtrTempB = srcPtrTempG + imageDim;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + 1;
            srcPtrTempB = srcPtrTempG + 1;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR += channel;
                srcPtrTempG += channel;
                srcPtrTempB += channel;
            }
        }
    }
    else if (channel == 1)
    {
        memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
    }

    Rpp32u newChannel = 1;

    RppiSize srcSizeMod, rppiKernelSize;
    Rpp32u kernelSize;
    int bound;

    // Sobel Filter

    kernelSize = 3;
    bound = (kernelSize - 1) / 2;

    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);

    T *dstPtrGreyscale = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));

    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
    generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;

    Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
    generate_sobel_kernel_host(kernelX, 1);
    Rpp32s *dstPtrIntermediateX = (Rpp32s *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32s));
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateX, srcSize, kernelX, rppiKernelSize, chnFormat, newChannel);

    Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
    generate_sobel_kernel_host(kernelY, 2);
    Rpp32s *dstPtrIntermediateY = (Rpp32s *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32s));
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateY, srcSize, kernelY, rppiKernelSize, chnFormat, newChannel);

    compute_magnitude_host(dstPtrIntermediateX, dstPtrIntermediateY, srcSize, dstPtrGreyscale, chnFormat, newChannel);

    // Find Image Maximum

    T *srcPtrTemp;
    srcPtrTemp = srcPtrGreyscale;
    Rpp8u max = *srcPtrTemp;
    for (int i = 0; i < (newChannel * srcSize.height * srcSize.width); i++)
    {
        if (*srcPtrTemp > max)
        {
            max = *srcPtrTemp;
        }
        srcPtrTemp++;
    }

    // Determine Gradients, Perform NMS, Double Thresholding and Edge Tracing by hysterisis

    Rpp32f gradient;
    Rpp32s *dstPtrIntermediateXTemp, *dstPtrIntermediateYTemp;
    dstPtrIntermediateXTemp = dstPtrIntermediateX;
    dstPtrIntermediateYTemp = dstPtrIntermediateY;

    T *srcPtrWindow, *dstPtrGreyscaleTemp, *srcPtrWindowCenter;
    srcPtrWindow = srcPtrMod;
    dstPtrGreyscaleTemp = dstPtrGreyscale;

    generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

    srcPtrWindowCenter = srcPtrWindow + (bound * srcSizeMod.width) + bound;
    Rpp32u toNeighborhood1 = 1;
    Rpp32u toNeighborhood2 = 2;
    Rpp32u toNeighborhood3 = srcSizeMod.width + 2;
    Rpp32u toNeighborhood4 = 2 * (srcSizeMod.width + 1);
    Rpp32u toNeighborhood5 = (2 * srcSizeMod.width) + 1;
    Rpp32u toNeighborhood6 = 2 * srcSizeMod.width;
    Rpp32u toNeighborhood7 = srcSizeMod.width;
    T *position1Ptr, *position2Ptr;
    dstPtrIntermediateXTemp = dstPtrIntermediateX;
    dstPtrIntermediateYTemp = dstPtrIntermediateY;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            gradient = atan((Rpp32f) *dstPtrIntermediateYTemp / (Rpp32f) *dstPtrIntermediateXTemp);
            if (gradient > 1.178097 || gradient < -1.178097)
            {
                position1Ptr = srcPtrWindow + toNeighborhood1;
                position2Ptr = srcPtrWindow + toNeighborhood5;
            }
            else if (gradient > 0.392699)
            {
                position1Ptr = srcPtrWindow + toNeighborhood2;
                position2Ptr = srcPtrWindow + toNeighborhood6;
            }
            else if (gradient < -0.392699)
            {
                position1Ptr = srcPtrWindow;
                position2Ptr = srcPtrWindow + toNeighborhood4;
            }
            else
            {
                position1Ptr = srcPtrWindow + toNeighborhood3;
                position2Ptr = srcPtrWindow + toNeighborhood7;
            }

            canny_non_max_suppression_kernel_host(dstPtrGreyscaleTemp, *srcPtrWindowCenter, position1Ptr, position2Ptr);

            if (*dstPtrGreyscaleTemp > maxThreshold)
            {
                *dstPtrGreyscaleTemp = (T) 255;
            }
            else if (*dstPtrGreyscaleTemp < minThreshold)
            {
                *dstPtrGreyscaleTemp = (T) 0;
            }
            else
            {
                *dstPtrGreyscaleTemp = (T) 100;
            }

            srcPtrWindow++;
            srcPtrWindowCenter++;
            dstPtrGreyscaleTemp++;
            dstPtrIntermediateXTemp++;
            dstPtrIntermediateYTemp++;
        }
        srcPtrWindow += (kernelSize - 1);
        srcPtrWindowCenter += (kernelSize - 1);
    }

    srcPtrWindow = srcPtrMod;
    dstPtrGreyscaleTemp = dstPtrGreyscale;
    generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

    srcPtrWindowCenter = srcPtrWindow + (bound * srcSizeMod.width) + bound;
    Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            if (*srcPtrWindowCenter == (T) 100)
            {
                canny_hysterisis_edge_tracing_kernel_host(srcPtrWindow, dstPtrGreyscaleTemp, srcSize,
                                kernelSize, remainingElementsInRow, *srcPtrWindowCenter, bound,
                                chnFormat, newChannel);
            }
            srcPtrWindow++;
            srcPtrWindowCenter++;
            dstPtrGreyscaleTemp++;
        }
        srcPtrWindow += (kernelSize - 1);
        srcPtrWindowCenter += (kernelSize - 1);
    }

    // Greyscale TO RGB Conversion

    dstPtrGreyscaleTemp = dstPtrGreyscale;
    T *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            for (int c = 0; c < channel; c++)
            {
                memcpy(dstPtrTemp, dstPtrGreyscaleTemp, imageDim * sizeof(T));
                dstPtrTemp += imageDim;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            for (int i = 0; i < imageDim; i++)
            {
                memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                dstPtrTemp++;
                memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                dstPtrTemp++;
                memcpy(dstPtrTemp, dstPtrGreyscaleTemp, sizeof(T));
                dstPtrTemp++;
                dstPtrGreyscaleTemp++;
            }
        }
    }
    else if (channel == 1)
    {
        memcpy(dstPtr, dstPtrGreyscale, imageDim * sizeof(T));
    }

    free(srcPtrGreyscale);
    free(dstPtrGreyscale);
    free(srcPtrMod);
    free(kernelX);
    free(dstPtrIntermediateX);
    free(kernelY);
    free(dstPtrIntermediateY);

    return RPP_SUCCESS;
}

/**************** laplacian_image_pyramid ***************/

template <typename T>
RppStatus laplacian_image_pyramid_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* batch_dstPtr,
                                             Rpp32f *batch_stdDev, Rpp32u *batch_kernelSize,
                                             Rpp32u nbatchSize,
                                             RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32f stdDev = batch_stdDev[batchCount];
        Rpp32u kernelSize = batch_kernelSize[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
        int bound = ((kernelSize - 1) / 2);

        generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

        RppiSize srcSizeMod;
        srcSizeMod.width = srcSize.width + (2 * bound);
        srcSizeMod.height = srcSize.height + (2 * bound);
        T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

        generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

        RppiSize rppiKernelSize;
        rppiKernelSize.height = kernelSize;
        rppiKernelSize.width = kernelSize;
        T *srcPtrConvolved = (T *)calloc(channel * srcSize.height * srcSize.width, sizeof(T));
        convolve_image_host(srcPtrMod, srcSizeMod, srcPtrConvolved, srcSize, kernel, rppiKernelSize, chnFormat, channel);

        RppiSize srcSize1;
        srcSize1.height = (srcSize.height + 1) / 2;
        srcSize1.width = (srcSize.width + 1) / 2;
        T *srcPtr1 = (T *)calloc(channel * srcSize1.height * srcSize1.width, sizeof(T));
        compute_downsampled_image_host(srcPtrConvolved, srcSize, srcPtr1, srcSize1, chnFormat, channel);

        RppiSize srcSize1Mod;
        srcSize1Mod.width = srcSize1.width + (2 * bound);
        srcSize1Mod.height = srcSize1.height + (2 * bound);
        T *srcPtr1Mod = (T *)calloc(srcSize1Mod.height * srcSize1Mod.width * channel, sizeof(T));

        generate_evenly_padded_image_host(srcPtr1, srcSize1, srcPtr1Mod, srcSize1Mod, chnFormat, channel);

        T *srcPtr1Convolved = (T *)calloc(channel * srcSize1.height * srcSize1.width, sizeof(T));
        convolve_image_host(srcPtr1Mod, srcSize1Mod, srcPtr1Convolved, srcSize1, kernel, rppiKernelSize, chnFormat, channel);

        T *dstPtr = (T*) calloc(channel * srcSize1.height * srcSize1.width, sizeof(T));

        compute_subtract_host(srcPtr1, srcPtr1Convolved, srcSize1, dstPtr, channel);

        compute_padded_from_unpadded_host(dstPtr, srcSize1, batch_srcSizeMax[batchCount], batch_dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtr);
        free(kernel);
        free(srcPtrMod);
        free(srcPtrConvolved);
        free(srcPtr1);
        free(srcPtr1Mod);
        free(srcPtr1Convolved);
        free(dstPtr);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus laplacian_image_pyramid_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;
    T *srcPtrConvolved = (T *)calloc(channel * srcSize.height * srcSize.width, sizeof(T));
    convolve_image_host(srcPtrMod, srcSizeMod, srcPtrConvolved, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    RppiSize srcSize1;
    srcSize1.height = (srcSize.height + 1) / 2;
    srcSize1.width = (srcSize.width + 1) / 2;
    T *srcPtr1 = (T *)calloc(channel * srcSize1.height * srcSize1.width, sizeof(T));
    compute_downsampled_image_host(srcPtrConvolved, srcSize, srcPtr1, srcSize1, chnFormat, channel);

    RppiSize srcSize1Mod;
    srcSize1Mod.width = srcSize1.width + (2 * bound);
    srcSize1Mod.height = srcSize1.height + (2 * bound);
    T *srcPtr1Mod = (T *)calloc(srcSize1Mod.height * srcSize1Mod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr1, srcSize1, srcPtr1Mod, srcSize1Mod, chnFormat, channel);

    T *srcPtr1Convolved = (T *)calloc(channel * srcSize1.height * srcSize1.width, sizeof(T));
    convolve_image_host(srcPtr1Mod, srcSize1Mod, srcPtr1Convolved, srcSize1, kernel, rppiKernelSize, chnFormat, channel);

    compute_subtract_host(srcPtr1, srcPtr1Convolved, srcSize1, dstPtr, channel);

    free(kernel);
    free(srcPtrMod);
    free(srcPtrConvolved);
    free(srcPtr1);
    free(srcPtr1Mod);
    free(srcPtr1Convolved);

    return RPP_SUCCESS;
}

/**************** harris_corner_detector ***************/

template <typename T>
RppStatus harris_corner_detector_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* batch_dstPtr,
                                            Rpp32u *batch_gaussianKernelSize, Rpp32f *batch_stdDev,
                                            Rpp32u *batch_kernelSize, Rpp32f *batch_kValue, Rpp32f *batch_threshold,
                                            Rpp32u *batch_nonmaxKernelSize,
                                            Rpp32u nbatchSize,
                                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u gaussianKernelSize = batch_gaussianKernelSize[batchCount];
        Rpp32f stdDev = batch_stdDev[batchCount];
        Rpp32u kernelSize = batch_kernelSize[batchCount];
        Rpp32f kValue = batch_kValue[batchCount];
        Rpp32f threshold = batch_threshold[batchCount];
        Rpp32u nonmaxKernelSize = batch_nonmaxKernelSize[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        // RGB to Greyscale Conversion

        Rpp32u imageDim = srcSize.height * srcSize.width;
        Rpp32u twiceImageDim = 2 * imageDim;

        T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
        T *srcPtrGreyscaleTemp;
        srcPtrGreyscaleTemp = srcPtrGreyscale;

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PLANAR)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + imageDim;
                srcPtrTempB = srcPtrTempG + imageDim;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
            }
            else if (chnFormat == RPPI_CHN_PACKED)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + 1;
                srcPtrTempB = srcPtrTempG + 1;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR += channel;
                    srcPtrTempG += channel;
                    srcPtrTempB += channel;
                }
            }
        }
        else if (channel == 1)
        {
            memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
        }

        Rpp32u newChannel = 1;

        // Gaussian Filter

        Rpp32f *gaussianKernel = (Rpp32f *)calloc(gaussianKernelSize * gaussianKernelSize, sizeof(Rpp32f));
        int gaussianBound = ((gaussianKernelSize - 1) / 2);

        generate_gaussian_kernel_host(stdDev, gaussianKernel, gaussianKernelSize);

        RppiSize srcSizeMod;
        srcSizeMod.width = srcSize.width + (2 * gaussianBound);
        srcSizeMod.height = srcSize.height + (2 * gaussianBound);
        T *srcPtrGaussianPadded = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));

        generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrGaussianPadded, srcSizeMod, chnFormat, newChannel);

        RppiSize rppiGaussianKernelSize;
        rppiGaussianKernelSize.height = gaussianKernelSize;
        rppiGaussianKernelSize.width = gaussianKernelSize;
        convolve_image_host(srcPtrGaussianPadded, srcSizeMod, srcPtrGreyscale, srcSize, gaussianKernel, rppiGaussianKernelSize, chnFormat, newChannel);

        // Sobel Filter

        RppiSize rppiSobelKernelSize;
        Rpp32u sobelKernelSize = 3;
        int sobelBound = (sobelKernelSize - 1) / 2;

        srcSizeMod.width = srcSize.width + (2 * sobelBound);
        srcSizeMod.height = srcSize.height + (2 * sobelBound);

        T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
        generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

        rppiSobelKernelSize.height = sobelKernelSize;
        rppiSobelKernelSize.width = sobelKernelSize;

        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        T *srcPtrDerivativeX = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
        convolve_image_host(srcPtrMod, srcSizeMod, srcPtrDerivativeX, srcSize, kernelX, rppiSobelKernelSize, chnFormat, newChannel);

        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        T *srcPtrDerivativeY = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
        convolve_image_host(srcPtrMod, srcSizeMod, srcPtrDerivativeY, srcSize, kernelY, rppiSobelKernelSize, chnFormat, newChannel);

        // Pad x and y gradient images

        int bound = (kernelSize - 1) / 2;
        RppiSize srcSizeDerivativeMod;
        srcSizeDerivativeMod.height = srcSize.height + (2 * bound);
        srcSizeDerivativeMod.width = srcSize.width + (2 * bound);

        T *srcPtrDerivativeXmod = (T *)calloc(srcSizeDerivativeMod.height * srcSizeDerivativeMod.width * newChannel, sizeof(T));
        generate_evenly_padded_image_host(srcPtrDerivativeX, srcSize, srcPtrDerivativeXmod, srcSizeDerivativeMod, chnFormat, newChannel);

        T *srcPtrDerivativeYmod = (T *)calloc(srcSizeDerivativeMod.height * srcSizeDerivativeMod.width * newChannel, sizeof(T));
        generate_evenly_padded_image_host(srcPtrDerivativeY, srcSize, srcPtrDerivativeYmod, srcSizeDerivativeMod, chnFormat, newChannel);

        // Compute the harris corner strengh matrix

        Rpp32f *dstPtrGreyscaleFloat = (Rpp32f *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32f));
        Rpp32f *dstPtrGreyscaleFloatTemp;
        dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat;

        T *srcPtrWindowX, *srcPtrWindowY;
        srcPtrWindowX = srcPtrDerivativeXmod;
        srcPtrWindowY = srcPtrDerivativeYmod;

        Rpp32u remainingElementsInRow = srcSizeDerivativeMod.width - kernelSize;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                harris_corner_detector_kernel_host(srcPtrWindowX, srcPtrWindowY, dstPtrGreyscaleFloatTemp, srcSize,
                                                    kernelSize, remainingElementsInRow, kValue, threshold,
                                                    chnFormat, channel);
                srcPtrWindowX++;
                srcPtrWindowY++;
                dstPtrGreyscaleFloatTemp++;
            }
            srcPtrWindowX += (kernelSize - 1);
            srcPtrWindowY += (kernelSize - 1);
        }

        int nonmaxBound = (nonmaxKernelSize - 1) / 2;
        RppiSize srcSizeNonmaxMod;
        srcSizeNonmaxMod.height = srcSize.height + (2 * nonmaxBound);
        srcSizeNonmaxMod.width = srcSize.width + (2 * nonmaxBound);

        Rpp32f *dstPtrGreyscaleFloatMod = (Rpp32f *)calloc(srcSizeNonmaxMod.height * srcSizeNonmaxMod.width * newChannel, sizeof(Rpp32f));
        generate_evenly_padded_image_host(dstPtrGreyscaleFloat, srcSize, dstPtrGreyscaleFloatMod, srcSizeNonmaxMod, chnFormat, newChannel);

        Rpp32f *dstPtrGreyscaleWindow;
        Rpp32f windowCenter;
        dstPtrGreyscaleWindow = dstPtrGreyscaleFloatMod;
        dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat;

        Rpp32u windowCenterPosIncrement = (nonmaxBound * srcSizeNonmaxMod.width) + nonmaxBound;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                windowCenter = (Rpp32f) *(dstPtrGreyscaleWindow + windowCenterPosIncrement);
                non_max_suppression_kernel_host(dstPtrGreyscaleWindow, dstPtrGreyscaleFloatTemp, srcSize,
                                    nonmaxKernelSize, remainingElementsInRow, windowCenter,
                                    RPPI_CHN_PLANAR, newChannel);
                dstPtrGreyscaleWindow++;
                dstPtrGreyscaleFloatTemp++;
            }
            dstPtrGreyscaleWindow += (nonmaxKernelSize - 1);
        }

        // Overlay Harris Corners on original image

        memcpy(dstPtr, srcPtr, channel * imageDim * sizeof(T));

        T *dstPtrWindow;

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat + (bound * srcSize.width) + bound;
            dstPtrWindow = dstPtr;
            Rpp32u remainingElementsInRow = srcSize.width - kernelSize;
            for (int i = (2 * bound); i < srcSize.height; i++)
            {
                for (int j = (2 * bound); j < srcSize.width; j++)
                {
                    if (*dstPtrGreyscaleFloatTemp != 0)
                    {
                        if (channel == 3)
                        {
                            harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_minimum_kernel_host(dstPtrWindow + imageDim, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_maximum_kernel_host(dstPtrWindow + twiceImageDim, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                        else if (channel == 1)
                        {
                            harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                    }
                    dstPtrGreyscaleFloatTemp++;
                    dstPtrWindow++;
                }
                dstPtrGreyscaleFloatTemp += (kernelSize - 1);
                dstPtrWindow += (kernelSize - 1);
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat + (newChannel * ((bound * srcSize.width) + bound));
            dstPtrWindow = dstPtr;
            Rpp32u remainingElementsInRow = channel * (srcSize.width - kernelSize);
            Rpp32u increment = channel * (kernelSize - 1);
            for (int i = (2 * bound); i < srcSize.height; i++)
            {
                for (int j = (2 * bound); j < srcSize.width; j++)
                {
                    if (*dstPtrGreyscaleFloatTemp != 0)
                    {
                        if (channel == 3)
                        {
                            harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_minimum_kernel_host(dstPtrWindow + 1, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_maximum_kernel_host(dstPtrWindow + 2, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                        else if (channel == 1)
                        {
                            harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                    }
                    dstPtrGreyscaleFloatTemp++;
                    dstPtrWindow += channel;
                }
                dstPtrGreyscaleFloatTemp += (kernelSize - 1);
                dstPtrWindow += increment;
            }
        }

        compute_padded_from_unpadded_host(dstPtr, srcSize, batch_srcSizeMax[batchCount], batch_dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtr);
        free(dstPtr);
        free(srcPtrGreyscale);
        free(gaussianKernel);
        free(srcPtrGaussianPadded);
        free(srcPtrMod);
        free(kernelX);
        free(srcPtrDerivativeX);
        free(kernelY);
        free(srcPtrDerivativeY);
        free(srcPtrDerivativeXmod);
        free(srcPtrDerivativeYmod);
        free(dstPtrGreyscaleFloat);
        free(dstPtrGreyscaleFloatMod);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus harris_corner_detector_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                      Rpp32u gaussianKernelSize, Rpp32f stdDev,
                                      Rpp32u kernelSize, Rpp32f kValue, Rpp32f threshold,
                                      Rpp32u nonmaxKernelSize,
                                      RppiChnFormat chnFormat, Rpp32u channel)
{
    // RGB to Greyscale Conversion

    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u twiceImageDim = 2 * imageDim;

    T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
    T *srcPtrGreyscaleTemp;
    srcPtrGreyscaleTemp = srcPtrGreyscale;

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + imageDim;
            srcPtrTempB = srcPtrTempG + imageDim;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + 1;
            srcPtrTempB = srcPtrTempG + 1;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR += channel;
                srcPtrTempG += channel;
                srcPtrTempB += channel;
            }
        }
    }
    else if (channel == 1)
    {
        memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
    }

    Rpp32u newChannel = 1;

    // Gaussian Filter

    Rpp32f *gaussianKernel = (Rpp32f *)calloc(gaussianKernelSize * gaussianKernelSize, sizeof(Rpp32f));
    int gaussianBound = ((gaussianKernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, gaussianKernel, gaussianKernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * gaussianBound);
    srcSizeMod.height = srcSize.height + (2 * gaussianBound);
    T *srcPtrGaussianPadded = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));

    generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrGaussianPadded, srcSizeMod, chnFormat, newChannel);

    RppiSize rppiGaussianKernelSize;
    rppiGaussianKernelSize.height = gaussianKernelSize;
    rppiGaussianKernelSize.width = gaussianKernelSize;
    convolve_image_host(srcPtrGaussianPadded, srcSizeMod, srcPtrGreyscale, srcSize, gaussianKernel, rppiGaussianKernelSize, chnFormat, newChannel);

    // Sobel Filter

    RppiSize rppiSobelKernelSize;
    Rpp32u sobelKernelSize = 3;
    int sobelBound = (sobelKernelSize - 1) / 2;

    srcSizeMod.width = srcSize.width + (2 * sobelBound);
    srcSizeMod.height = srcSize.height + (2 * sobelBound);

    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
    generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, chnFormat, newChannel);

    rppiSobelKernelSize.height = sobelKernelSize;
    rppiSobelKernelSize.width = sobelKernelSize;

    Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
    generate_sobel_kernel_host(kernelX, 1);
    T *srcPtrDerivativeX = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
    convolve_image_host(srcPtrMod, srcSizeMod, srcPtrDerivativeX, srcSize, kernelX, rppiSobelKernelSize, chnFormat, newChannel);

    Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
    generate_sobel_kernel_host(kernelY, 2);
    T *srcPtrDerivativeY = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
    convolve_image_host(srcPtrMod, srcSizeMod, srcPtrDerivativeY, srcSize, kernelY, rppiSobelKernelSize, chnFormat, newChannel);

    // Pad x and y gradient images

    int bound = (kernelSize - 1) / 2;
    RppiSize srcSizeDerivativeMod;
    srcSizeDerivativeMod.height = srcSize.height + (2 * bound);
    srcSizeDerivativeMod.width = srcSize.width + (2 * bound);

    T *srcPtrDerivativeXmod = (T *)calloc(srcSizeDerivativeMod.height * srcSizeDerivativeMod.width * newChannel, sizeof(T));
    generate_evenly_padded_image_host(srcPtrDerivativeX, srcSize, srcPtrDerivativeXmod, srcSizeDerivativeMod, chnFormat, newChannel);

    T *srcPtrDerivativeYmod = (T *)calloc(srcSizeDerivativeMod.height * srcSizeDerivativeMod.width * newChannel, sizeof(T));
    generate_evenly_padded_image_host(srcPtrDerivativeY, srcSize, srcPtrDerivativeYmod, srcSizeDerivativeMod, chnFormat, newChannel);

    // Compute the harris corner strengh matrix

    Rpp32f *dstPtrGreyscaleFloat = (Rpp32f *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32f));
    Rpp32f *dstPtrGreyscaleFloatTemp;
    dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat;

    T *srcPtrWindowX, *srcPtrWindowY;
    srcPtrWindowX = srcPtrDerivativeXmod;
    srcPtrWindowY = srcPtrDerivativeYmod;

    Rpp32u remainingElementsInRow = srcSizeDerivativeMod.width - kernelSize;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            harris_corner_detector_kernel_host(srcPtrWindowX, srcPtrWindowY, dstPtrGreyscaleFloatTemp, srcSize,
                                                kernelSize, remainingElementsInRow, kValue, threshold,
                                                chnFormat, channel);
            srcPtrWindowX++;
            srcPtrWindowY++;
            dstPtrGreyscaleFloatTemp++;
        }
        srcPtrWindowX += (kernelSize - 1);
        srcPtrWindowY += (kernelSize - 1);
    }

    int nonmaxBound = (nonmaxKernelSize - 1) / 2;
    RppiSize srcSizeNonmaxMod;
    srcSizeNonmaxMod.height = srcSize.height + (2 * nonmaxBound);
    srcSizeNonmaxMod.width = srcSize.width + (2 * nonmaxBound);

    Rpp32f *dstPtrGreyscaleFloatMod = (Rpp32f *)calloc(srcSizeNonmaxMod.height * srcSizeNonmaxMod.width * newChannel, sizeof(Rpp32f));
    generate_evenly_padded_image_host(dstPtrGreyscaleFloat, srcSize, dstPtrGreyscaleFloatMod, srcSizeNonmaxMod, chnFormat, newChannel);

    Rpp32f *dstPtrGreyscaleWindow;
    Rpp32f windowCenter;
    dstPtrGreyscaleWindow = dstPtrGreyscaleFloatMod;
    dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat;

    Rpp32u windowCenterPosIncrement = (nonmaxBound * srcSizeNonmaxMod.width) + nonmaxBound;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            windowCenter = (Rpp32f) *(dstPtrGreyscaleWindow + windowCenterPosIncrement);
            non_max_suppression_kernel_host(dstPtrGreyscaleWindow, dstPtrGreyscaleFloatTemp, srcSize,
                                nonmaxKernelSize, remainingElementsInRow, windowCenter,
                                RPPI_CHN_PLANAR, newChannel);
            dstPtrGreyscaleWindow++;
            dstPtrGreyscaleFloatTemp++;
        }
        dstPtrGreyscaleWindow += (nonmaxKernelSize - 1);
    }

    // Overlay Harris Corners on original image

    memcpy(dstPtr, srcPtr, channel * imageDim * sizeof(T));

    T *dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat + (bound * srcSize.width) + bound;
        dstPtrWindow = dstPtr;
        Rpp32u remainingElementsInRow = srcSize.width - kernelSize;
        for (int i = (2 * bound); i < srcSize.height; i++)
        {
            for (int j = (2 * bound); j < srcSize.width; j++)
            {
                if (*dstPtrGreyscaleFloatTemp != 0)
                {
                    if (channel == 3)
                    {
                        harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_minimum_kernel_host(dstPtrWindow + imageDim, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_maximum_kernel_host(dstPtrWindow + twiceImageDim, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                    else if (channel == 1)
                    {
                        harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                }
                dstPtrGreyscaleFloatTemp++;
                dstPtrWindow++;
            }
            dstPtrGreyscaleFloatTemp += (kernelSize - 1);
            dstPtrWindow += (kernelSize - 1);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        dstPtrGreyscaleFloatTemp = dstPtrGreyscaleFloat + (channel * ((bound * srcSize.width) + bound));
        dstPtrWindow = dstPtr;
        Rpp32u remainingElementsInRow = channel * (srcSize.width - kernelSize);
        Rpp32u increment = channel * (kernelSize - 1);
        for (int i = (2 * bound); i < srcSize.height; i++)
        {
            for (int j = (2 * bound); j < srcSize.width; j++)
            {
                if (*dstPtrGreyscaleFloatTemp != 0)
                {
                    if (channel == 3)
                    {
                        harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_minimum_kernel_host(dstPtrWindow + 1, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_maximum_kernel_host(dstPtrWindow + 2, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                    else if (channel == 1)
                    {
                        harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                }
                dstPtrGreyscaleFloatTemp++;
                dstPtrWindow += channel;
            }
            dstPtrGreyscaleFloatTemp += (kernelSize - 1);
            dstPtrWindow += increment;
        }
    }

    free(srcPtrGreyscale);
    free(gaussianKernel);
    free(srcPtrGaussianPadded);
    free(srcPtrMod);
    free(kernelX);
    free(srcPtrDerivativeX);
    free(kernelY);
    free(srcPtrDerivativeY);
    free(srcPtrDerivativeXmod);
    free(srcPtrDerivativeYmod);
    free(dstPtrGreyscaleFloat);
    free(dstPtrGreyscaleFloatMod);

    return RPP_SUCCESS;
}


/**************** reconstruction_laplacian_image_pyramid ***************/

template <typename T>
RppStatus reconstruction_laplacian_image_pyramid_host_batch(T* batch_srcPtr1, RppiSize *batch_srcSize1, RppiSize *batch_srcSizeMax1,
                                                            T* batch_srcPtr2, RppiSize *batch_srcSize2, RppiSize *batch_srcSizeMax2,
                                                            T* batch_dstPtr,
                                                            Rpp32f *batch_stdDev, Rpp32u *batch_kernelSize,
                                                            Rpp32u nbatchSize,
                                                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32f stdDev = batch_stdDev[batchCount];
        Rpp32u kernelSize = batch_kernelSize[batchCount];

        Rpp32u loc1 = 0;
        compute_image_location_host(batch_srcSizeMax1, batchCount, &loc1, channel);
        Rpp32u loc2 = 0;
        compute_image_location_host(batch_srcSizeMax2, batchCount, &loc2, channel);

        T *srcPtr1 = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));
        T *srcPtr2 = (T*) calloc(channel * batch_srcSize2[batchCount].height * batch_srcSize2[batchCount].width, sizeof(T));
        T *dstPtr = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr1 + loc1, batch_srcSize1[batchCount], batch_srcSizeMax1[batchCount], srcPtr1,
                                          chnFormat, channel);
        compute_unpadded_from_padded_host(batch_srcPtr2 + loc2, batch_srcSize2[batchCount], batch_srcSizeMax2[batchCount], srcPtr2,
                                          chnFormat, channel);

        RppiSize srcSize1, srcSize2;
        srcSize1.height = batch_srcSize1[batchCount].height;
        srcSize1.width = batch_srcSize1[batchCount].width;
        srcSize2.height = batch_srcSize2[batchCount].height;
        srcSize2.width = batch_srcSize2[batchCount].width;

        T *srcPtr2Upsampled = (T *)calloc(srcSize1.height * srcSize1.width * channel, sizeof(T));

        resize_kernel_host(srcPtr2, srcSize2, srcPtr2Upsampled, srcSize1, chnFormat, channel);

        Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
        int bound = ((kernelSize - 1) / 2);

        generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

        RppiSize srcSize1Mod;
        srcSize1Mod.width = srcSize1.width + (2 * bound);
        srcSize1Mod.height = srcSize1.height + (2 * bound);
        T *srcPtr2UpsampledMod = (T *)calloc(srcSize1Mod.height * srcSize1Mod.width * channel, sizeof(T));

        generate_evenly_padded_image_host(srcPtr2Upsampled, srcSize1, srcPtr2UpsampledMod, srcSize1Mod, chnFormat, channel);

        RppiSize rppiKernelSize;
        rppiKernelSize.height = kernelSize;
        rppiKernelSize.width = kernelSize;
        convolve_image_host(srcPtr2UpsampledMod, srcSize1Mod, dstPtr, srcSize1, kernel, rppiKernelSize, chnFormat, channel);

        accumulate_kernel_host(dstPtr, srcPtr1, srcSize1, chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtr, srcSize1, batch_srcSizeMax1[batchCount], batch_dstPtr + loc1,
                                          chnFormat, channel);

        free(srcPtr1);
        free(srcPtr2);
        free(dstPtr);
        free(srcPtr2Upsampled);
        free(kernel);
        free(srcPtr2UpsampledMod);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus reconstruction_laplacian_image_pyramid_host(T* srcPtr1, RppiSize srcSize1, T* srcPtr2, RppiSize srcSize2, T* dstPtr,
                                                      Rpp32f stdDev, Rpp32u kernelSize,
                                                      RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr2Upsampled = (T *)calloc(srcSize1.height * srcSize1.width * channel, sizeof(T));

    resize_kernel_host(srcPtr2, srcSize2, srcPtr2Upsampled, srcSize1, chnFormat, channel);

    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

    RppiSize srcSize1Mod;
    srcSize1Mod.width = srcSize1.width + (2 * bound);
    srcSize1Mod.height = srcSize1.height + (2 * bound);
    T *srcPtr2UpsampledMod = (T *)calloc(srcSize1Mod.height * srcSize1Mod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr2Upsampled, srcSize1, srcPtr2UpsampledMod, srcSize1Mod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;
    convolve_image_host(srcPtr2UpsampledMod, srcSize1Mod, dstPtr, srcSize1, kernel, rppiKernelSize, chnFormat, channel);

    accumulate_kernel_host(dstPtr, srcPtr1, srcSize1, chnFormat, channel);

    free(srcPtr2Upsampled);
    free(kernel);
    free(srcPtr2UpsampledMod);

    return RPP_SUCCESS;
}


/**************** hough_lines ***************/

template <typename T, typename U>
RppStatus hough_lines_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* batch_lines,
                                 Rpp32f *batch_rho, Rpp32f *batch_theta, Rpp32u *batch_threshold,
                                 Rpp32u *batch_lineLength, Rpp32u *batch_lineGap, Rpp32u *batch_linesMax,
                                 Rpp32u nbatchSize,
                                 RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32f rho = batch_rho[batchCount];
        Rpp32f theta = batch_theta[batchCount];
        Rpp32u threshold = batch_threshold[batchCount];
        Rpp32u lineLength = batch_lineLength[batchCount];
        Rpp32u lineGap = batch_lineGap[batchCount];
        Rpp32u linesMax = batch_linesMax[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        U *lines;
        lines = batch_lines + (batchCount * 4 * linesMax);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        // Initializations

        srand (time(NULL));

        U *linesTemp;
        linesTemp = lines;
        Rpp32u numofLines = 0;

        Rpp8u *srcPtrTemp;
        srcPtrTemp = srcPtr;

        RppiSize accumulatorSize;
        accumulatorSize.width = round(((srcSize.width + srcSize.height) * 2 + 1) / rho);
        accumulatorSize.height = round(PI / theta);

        Rpp32s *accumulator = (Rpp32s*)calloc(accumulatorSize.width * accumulatorSize.height, sizeof(Rpp32s));
        Rpp32s *accumulatorTemp, *accumulatorTemp2;

        T *validPixelMask = (T*)calloc(srcSize.height * srcSize.width, sizeof(T));
        T *validPixelMaskTemp;
        validPixelMaskTemp = validPixelMask;

        Rpp32f *cosLookUpTable = (Rpp32f*)calloc(accumulatorSize.height, sizeof(Rpp32f));
        Rpp32f *sinLookUpTable = (Rpp32f*)calloc(accumulatorSize.height, sizeof(Rpp32f));
        Rpp32f *cosLookUpTableTemp, *sinLookUpTableTemp;
        cosLookUpTableTemp = cosLookUpTable;
        sinLookUpTableTemp = sinLookUpTable;

        for(int n = 0; n < accumulatorSize.height; n++ )
        {
            *cosLookUpTableTemp = (Rpp32f) (cos((Rpp64f) (n * theta)) / rho);
            *sinLookUpTableTemp = (Rpp32f) (sin((Rpp64f) (n * theta)) / rho);
            cosLookUpTableTemp++;
            sinLookUpTableTemp++;
        }
        std::vector<Rpp32u> validPixelLocations;

        // Valid pixel locations
        for( int i = 0; i < srcSize.height; i++ )
        {
            for( int j = 0; j < srcSize.width; j++ )
            {
                if(*srcPtrTemp)
                {
                    *validPixelMaskTemp = (T) 1;
                    validPixelLocations.push_back(j);
                    validPixelLocations.push_back(i);
                }
                else
                {
                    *validPixelMaskTemp = (T) 0;
                }
                srcPtrTemp++;
                validPixelMaskTemp++;
            }
        }

        int count = (int)validPixelLocations.size() / 2;
        Rpp32s *endpoints = (Rpp32s*)calloc(4, sizeof(Rpp32s));

        // Process random pixels
        for( ; count > 0; count-- )
        {
            // Random point
            Rpp32u randomPixelLoc = rand() % (count + 1);
            int max_val = threshold-1, max_n = 0;
            Rpp32u pixelLocX, pixelLocY;
            pixelLocX = validPixelLocations[2 * randomPixelLoc];
            pixelLocY = validPixelLocations[(2 * randomPixelLoc) + 1];

            Rpp32f a, b;
            accumulatorTemp = accumulator;
            Rpp32s i = pixelLocY, j = pixelLocX, k, x0, y0, dx0, dy0, xflag;
            Rpp32s good_line;
            Rpp32s shift = 16;

            // Remove it by overriding it with the last element
            validPixelLocations[2 * randomPixelLoc] = validPixelLocations[2 * (count-1)];
            validPixelLocations[(2 * randomPixelLoc) + 1] = validPixelLocations[(2 * (count-1)) + 1];

            // Check if pixel is part of some other line
            if(!*(validPixelMask + (i * srcSize.width) + j))
                continue;

            // Update the accumulator
            cosLookUpTableTemp = cosLookUpTable;
            sinLookUpTableTemp = sinLookUpTable;
            for( int n = 0; n < accumulatorSize.height; n++, accumulatorTemp += accumulatorSize.width )
            {
                Rpp32s r = round( j * *cosLookUpTableTemp + i * *sinLookUpTableTemp );
                r += (accumulatorSize.width - 1) / 2;
                *(accumulatorTemp + r) += 1;
                int val = *(accumulatorTemp + r);
                if( max_val < val )
                {
                    max_val = val;
                    max_n = n;
                }
                cosLookUpTableTemp++;
                sinLookUpTableTemp++;
            }

            // Compare against threshold given by the user
            if( max_val < threshold )
                continue;

            // From the selected point, walk in each direction along the found line and extract line segment
            a = -*(sinLookUpTable + max_n);
            b = *(cosLookUpTable + max_n);
            x0 = j;
            y0 = i;
            if( RPPABS(a) > RPPABS(b) )
            {
                xflag = 1;
                dx0 = a > 0 ? 1 : -1;
                dy0 = round( b*(1 << shift)/RPPABS(a) );
                y0 = (y0 << shift) + (1 << (shift-1));
            }
            else
            {
                xflag = 0;
                dy0 = b > 0 ? 1 : -1;
                dx0 = round( a*(1 << shift)/RPPABS(b) );
                x0 = (x0 << shift) + (1 << (shift-1));
            }

            for( k = 0; k < 4; k += 2 )
            {
                Rpp32s gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

                if( k > 0 )
                    dx = -dx, dy = -dy;

                // Walk along the line, Stop at either the image border or in case of too big gap
                for( ;; x += dx, y += dy )
                {
                    Rpp32s i1, j1;

                    if( xflag )
                    {
                        j1 = x;
                        i1 = y >> shift;
                    }
                    else
                    {
                        j1 = x >> shift;
                        i1 = y;
                    }

                    if( j1 < 0 || j1 >= srcSize.width || i1 < 0 || i1 >= srcSize.height )
                        break;

                    validPixelMaskTemp = validPixelMask + (i1 * srcSize.width) + j1;

                    // For all valid points update line end, clear the mask element and reset the gap
                    if(*validPixelMaskTemp)
                    {
                        gap = 0;
                        *(endpoints + k + 1) = i1;
                        *(endpoints + k) = j1;
                    }
                    else if( ++gap > lineGap )
                        break;
                }
            }

            good_line = RPPABS(*(endpoints + 2) - *(endpoints + 0)) >= lineLength ||
                        RPPABS(*(endpoints + 3) - *(endpoints + 1)) >= lineLength;

            for( k = 0; k < 4; k += 2 )
            {
                int x = x0, y = y0, dx = dx0, dy = dy0;

                if( k > 0 )
                    dx = -dx, dy = -dy;

                // Walk along the line, Stop at either the image border or in case of too big gap
                for( ;; x += dx, y += dy )
                {
                    Rpp32s i1, j1;

                    if( xflag )
                    {
                        j1 = x;
                        i1 = y >> shift;
                    }
                    else
                    {
                        j1 = x >> shift;
                        i1 = y;
                    }

                    validPixelMaskTemp = validPixelMask + (i1 * srcSize.width) + j1;

                    // For all valid points update line end, clear the mask element and reset the gap
                    if(*validPixelMaskTemp)
                    {
                        if( good_line )
                        {
                            accumulatorTemp2 = accumulator;
                            cosLookUpTableTemp = cosLookUpTable;
                            sinLookUpTableTemp = sinLookUpTable;
                            for( int n = 0; n < accumulatorSize.height; n++, accumulatorTemp2 += accumulatorSize.width )
                            {
                                Rpp32s r = round( j1 * *cosLookUpTableTemp + i1 * *sinLookUpTableTemp);
                                r += (accumulatorSize.width - 1) / 2;
                                *(accumulatorTemp2 + r) -= 1;
                                cosLookUpTableTemp++;
                                sinLookUpTableTemp++;
                            }
                        }
                        *validPixelMaskTemp = (T) 0;
                    }

                    if( i1 == *(endpoints + k + 1) && j1 == *(endpoints + k) )
                        break;
                }
            }

            if( good_line )
            {
                *linesTemp = (U) *(endpoints + 0);
                linesTemp++;
                *linesTemp = (U) *(endpoints + 1);
                linesTemp++;
                *linesTemp = (U) *(endpoints + 2);
                linesTemp++;
                *linesTemp = (U) *(endpoints + 3);
                linesTemp++;

                numofLines++;
                if(numofLines >= linesMax)
                    // return RPP_SUCCESS;
                    break;
            }
        }

        free(srcPtr);
        free(accumulator);
        free(validPixelMask);
        free(cosLookUpTable);
        free(sinLookUpTable);
        free(endpoints);
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus hough_lines_host(T* srcPtr, RppiSize srcSize, U* lines,
                           Rpp32f rho, Rpp32f theta, Rpp32u threshold,
                           Rpp32u lineLength, Rpp32u lineGap, Rpp32u linesMax)
{
    // Initializations

    srand (time(NULL));

    U *linesTemp;
    linesTemp = lines;
    Rpp32u numofLines = 0;

    Rpp8u *srcPtrTemp;
    srcPtrTemp = srcPtr;

    RppiSize accumulatorSize;
    accumulatorSize.width = round(((srcSize.width + srcSize.height) * 2 + 1) / rho);
    accumulatorSize.height = round(PI / theta);

    Rpp32s *accumulator = (Rpp32s*)calloc(accumulatorSize.width * accumulatorSize.height, sizeof(Rpp32s));
    Rpp32s *accumulatorTemp, *accumulatorTemp2;

    T *validPixelMask = (T*)calloc(srcSize.height * srcSize.width, sizeof(T));
    T *validPixelMaskTemp;
    validPixelMaskTemp = validPixelMask;

    Rpp32f *cosLookUpTable = (Rpp32f*)calloc(accumulatorSize.height, sizeof(Rpp32f));
    Rpp32f *sinLookUpTable = (Rpp32f*)calloc(accumulatorSize.height, sizeof(Rpp32f));
    Rpp32f *cosLookUpTableTemp, *sinLookUpTableTemp;
    cosLookUpTableTemp = cosLookUpTable;
    sinLookUpTableTemp = sinLookUpTable;

    for(int n = 0; n < accumulatorSize.height; n++ )
    {
        *cosLookUpTableTemp = (Rpp32f) (cos((Rpp64f) (n * theta)) / rho);
        *sinLookUpTableTemp = (Rpp32f) (sin((Rpp64f) (n * theta)) / rho);
        cosLookUpTableTemp++;
        sinLookUpTableTemp++;
    }
    std::vector<Rpp32u> validPixelLocations;

    // Valid pixel locations
    for( int i = 0; i < srcSize.height; i++ )
    {
        for( int j = 0; j < srcSize.width; j++ )
        {
            if(*srcPtrTemp)
            {
                *validPixelMaskTemp = (T) 1;
                validPixelLocations.push_back(j);
                validPixelLocations.push_back(i);
            }
            else
            {
                *validPixelMaskTemp = (T) 0;
            }
            srcPtrTemp++;
            validPixelMaskTemp++;
        }
    }

    int count = (int)validPixelLocations.size() / 2;
    Rpp32s *endpoints = (Rpp32s*)calloc(4, sizeof(Rpp32s));

    // Process random pixels
    for( ; count > 0; count-- )
    {
        // Random point
        Rpp32u randomPixelLoc = rand() % (count + 1);
        int max_val = threshold-1, max_n = 0;
        Rpp32u pixelLocX, pixelLocY;
        pixelLocX = validPixelLocations[2 * randomPixelLoc];
        pixelLocY = validPixelLocations[(2 * randomPixelLoc) + 1];

        Rpp32f a, b;
        accumulatorTemp = accumulator;
        Rpp32s i = pixelLocY, j = pixelLocX, k, x0, y0, dx0, dy0, xflag;
        Rpp32s good_line;
        Rpp32s shift = 16;

        // Remove it by overriding it with the last element
        validPixelLocations[2 * randomPixelLoc] = validPixelLocations[2 * (count-1)];
        validPixelLocations[(2 * randomPixelLoc) + 1] = validPixelLocations[(2 * (count-1)) + 1];

        // Check if pixel is part of some other line
        if(!*(validPixelMask + (i * srcSize.width) + j))
            continue;

        // Update the accumulator
        cosLookUpTableTemp = cosLookUpTable;
        sinLookUpTableTemp = sinLookUpTable;
        for( int n = 0; n < accumulatorSize.height; n++, accumulatorTemp += accumulatorSize.width )
        {
            Rpp32s r = round( j * *cosLookUpTableTemp + i * *sinLookUpTableTemp );
            r += (accumulatorSize.width - 1) / 2;
            *(accumulatorTemp + r) += 1;
            int val = *(accumulatorTemp + r);
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
            cosLookUpTableTemp++;
            sinLookUpTableTemp++;
        }

        // Compare against threshold given by the user
        if( max_val < threshold )
            continue;

        // From the selected point, walk in each direction along the found line and extract line segment
        a = -*(sinLookUpTable + max_n);
        b = *(cosLookUpTable + max_n);
        x0 = j;
        y0 = i;
        if( RPPABS(a) > RPPABS(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = round( b*(1 << shift)/RPPABS(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = round( a*(1 << shift)/RPPABS(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 4; k += 2 )
        {
            Rpp32s gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // Walk along the line, Stop at either the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                Rpp32s i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= srcSize.width || i1 < 0 || i1 >= srcSize.height )
                    break;

                validPixelMaskTemp = validPixelMask + (i1 * srcSize.width) + j1;

                // For all valid points update line end, clear the mask element and reset the gap
                if(*validPixelMaskTemp)
                {
                    gap = 0;
                    *(endpoints + k + 1) = i1;
                    *(endpoints + k) = j1;
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = RPPABS(*(endpoints + 2) - *(endpoints + 0)) >= lineLength ||
                    RPPABS(*(endpoints + 3) - *(endpoints + 1)) >= lineLength;

        for( k = 0; k < 4; k += 2 )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // Walk along the line, Stop at either the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                Rpp32s i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                validPixelMaskTemp = validPixelMask + (i1 * srcSize.width) + j1;

                // For all valid points update line end, clear the mask element and reset the gap
                if(*validPixelMaskTemp)
                {
                    if( good_line )
                    {
                        accumulatorTemp2 = accumulator;
                        cosLookUpTableTemp = cosLookUpTable;
                        sinLookUpTableTemp = sinLookUpTable;
                        for( int n = 0; n < accumulatorSize.height; n++, accumulatorTemp2 += accumulatorSize.width )
                        {
                            Rpp32s r = round( j1 * *cosLookUpTableTemp + i1 * *sinLookUpTableTemp);
                            r += (accumulatorSize.width - 1) / 2;
                            *(accumulatorTemp2 + r) -= 1;
                            cosLookUpTableTemp++;
                            sinLookUpTableTemp++;
                        }
                    }
                    *validPixelMaskTemp = (T) 0;
                }

                if( i1 == *(endpoints + k + 1) && j1 == *(endpoints + k) )
                    break;
            }
        }

        if( good_line )
        {
            *linesTemp = (U) *(endpoints + 0);
            linesTemp++;
            *linesTemp = (U) *(endpoints + 1);
            linesTemp++;
            *linesTemp = (U) *(endpoints + 2);
            linesTemp++;
            *linesTemp = (U) *(endpoints + 3);
            linesTemp++;

            numofLines++;
            if(numofLines >= linesMax)
                return RPP_SUCCESS;
        }
    }

    free(accumulator);
    free(validPixelMask);
    free(cosLookUpTable);
    free(sinLookUpTable);
    free(endpoints);

    return RPP_SUCCESS;
}


/**************** fast_corner_detector ***************/

template <typename T>
RppStatus fast_corner_detector_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* batch_dstPtr,
                                          Rpp32u *batch_numOfPixels, T *batch_threshold,
                                          Rpp32u *batch_nonmaxKernelSize,
                                          Rpp32u nbatchSize,
                                          RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u numOfPixels = batch_numOfPixels[batchCount];
        T threshold = batch_threshold[batchCount];
        Rpp32u nonmaxKernelSize = batch_nonmaxKernelSize[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        // RGB to Greyscale Conversion

        Rpp32u imageDim = srcSize.height * srcSize.width;
        Rpp32u twiceImageDim = 2 * imageDim;

        T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
        T *srcPtrGreyscaleTemp;
        srcPtrGreyscaleTemp = srcPtrGreyscale;

        if (channel == 3)
        {
            if (chnFormat == RPPI_CHN_PLANAR)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + imageDim;
                srcPtrTempB = srcPtrTempG + imageDim;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
            }
            else if (chnFormat == RPPI_CHN_PACKED)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtr;
                srcPtrTempG = srcPtr + 1;
                srcPtrTempB = srcPtrTempG + 1;

                for (int i = 0; i < imageDim; i++)
                {
                    *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                    srcPtrGreyscaleTemp++;
                    srcPtrTempR += channel;
                    srcPtrTempG += channel;
                    srcPtrTempB += channel;
                }
            }
        }
        else if (channel == 1)
        {
            memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
        }

        Rpp32u newChannel = 1;

        // Pad image

        int bound = 3;
        RppiSize srcSizeMod;
        srcSizeMod.height = srcSize.height + (2 * bound);
        srcSizeMod.width = srcSize.width + (2 * bound);

        T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
        generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, RPPI_CHN_PLANAR, newChannel);

        // Compute the fast corner strengh matrix

        Rpp32u kernelSize = 7;
        T *dstPtrGreyscale = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
        T *dstPtrGreyscaleTemp;
        dstPtrGreyscaleTemp = dstPtrGreyscale;

        T *srcPtrWindow;
        srcPtrWindow = srcPtrMod;

        Rpp32u *bresenhamCirclePositions = (Rpp32u*) calloc(16, sizeof(Rpp32u));
        Rpp32u *bresenhamCirclePositionsTemp;
        bresenhamCirclePositionsTemp = bresenhamCirclePositions;

        *bresenhamCirclePositionsTemp = 3;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = 4;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (1 * srcSizeMod.width) + 5;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (2 * srcSizeMod.width) + 6;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (3 * srcSizeMod.width) + 6;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (4 * srcSizeMod.width) + 6;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (5 * srcSizeMod.width) + 5;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 4;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 3;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 2;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (5 * srcSizeMod.width) + 1;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (4 * srcSizeMod.width);
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (3 * srcSizeMod.width);
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (2 * srcSizeMod.width);
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = (1 * srcSizeMod.width) + 1;
        bresenhamCirclePositionsTemp++;
        *bresenhamCirclePositionsTemp = 2;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                fast_corner_detector_kernel_host(srcPtrWindow, dstPtrGreyscaleTemp, srcSizeMod,
                                                bresenhamCirclePositions, threshold, numOfPixels);
                srcPtrWindow++;
                dstPtrGreyscaleTemp++;
            }
            srcPtrWindow += (kernelSize - 1);
        }

        // Create score function

        generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, RPPI_CHN_PLANAR, newChannel);

        Rpp32u *dstPtrGreyscale32u = (Rpp32u *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32u));
        Rpp32u *dstPtrGreyscale32uTemp;
        Rpp32u windowCenter;

        srcPtrWindow = srcPtrMod;
        dstPtrGreyscale32uTemp = dstPtrGreyscale32u;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                windowCenter = (Rpp32u) *(srcPtrWindow + (3 * srcSizeMod.width) + 3);
                if (windowCenter != 0)
                {
                    fast_corner_detector_score_function_kernel_host(srcPtrWindow, dstPtrGreyscale32uTemp, srcSize, bresenhamCirclePositions, windowCenter);
                }
                else
                {
                    *dstPtrGreyscale32uTemp = (Rpp32u) 0;
                }
                srcPtrWindow++;
                dstPtrGreyscale32uTemp++;
            }
            srcPtrWindow += (kernelSize - 1);
        }

        // Apply non max suppression

        int nonmaxBound = (nonmaxKernelSize - 1) / 2;
        RppiSize srcSizeNonmaxMod;
        srcSizeNonmaxMod.height = srcSize.height + (2 * nonmaxBound);
        srcSizeNonmaxMod.width = srcSize.width + (2 * nonmaxBound);

        Rpp32u *dstPtrGreyscale32uMod = (Rpp32u *)calloc(srcSizeNonmaxMod.height * srcSizeNonmaxMod.width * newChannel, sizeof(Rpp32u));
        generate_evenly_padded_image_host(dstPtrGreyscale32u, srcSize, dstPtrGreyscale32uMod, srcSizeNonmaxMod, RPPI_CHN_PLANAR, newChannel);

        Rpp32u *dstPtrGreyscale32uWindow;
        dstPtrGreyscale32uWindow = dstPtrGreyscale32uMod;
        dstPtrGreyscale32uTemp = dstPtrGreyscale32u;

        Rpp32u windowCenterPosIncrement = (nonmaxBound * srcSizeNonmaxMod.width) + nonmaxBound;
        Rpp32u remainingElementsInRow = srcSizeNonmaxMod.width - nonmaxKernelSize;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                windowCenter = (Rpp32u) *(dstPtrGreyscale32uWindow + windowCenterPosIncrement);
                non_max_suppression_kernel_host(dstPtrGreyscale32uWindow, dstPtrGreyscale32uTemp, srcSize,
                                    nonmaxKernelSize, remainingElementsInRow, windowCenter,
                                    RPPI_CHN_PLANAR, newChannel);
                dstPtrGreyscale32uWindow++;
                dstPtrGreyscale32uTemp++;
            }
            dstPtrGreyscale32uWindow += (nonmaxKernelSize - 1);
        }

        // Overlay Fast Corners on original image - Large Dot

        memcpy(dstPtr, srcPtr, channel * imageDim * sizeof(T));

        T *dstPtrWindow;
        kernelSize = 3;
        bound = 1;

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            dstPtrGreyscale32uTemp = dstPtrGreyscale32u + (bound * srcSize.width) + bound;
            dstPtrWindow = dstPtr;
            Rpp32u remainingElementsInRow = srcSize.width - kernelSize;
            for (int i = (2 * bound); i < srcSize.height; i++)
            {
                for (int j = (2 * bound); j < srcSize.width; j++)
                {
                    if (*dstPtrGreyscale32uTemp != 0)
                    {
                        if (channel == 3)
                        {
                            harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_minimum_kernel_host(dstPtrWindow + imageDim, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_maximum_kernel_host(dstPtrWindow + twiceImageDim, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                        else if (channel == 1)
                        {
                            harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                    }
                    dstPtrGreyscale32uTemp++;
                    dstPtrWindow++;
                }
                dstPtrGreyscale32uTemp += (kernelSize - 1);
                dstPtrWindow += (kernelSize - 1);
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrGreyscale32uTemp = dstPtrGreyscale32u + (newChannel * ((bound * srcSize.width) + bound));
            dstPtrWindow = dstPtr;
            Rpp32u remainingElementsInRow = channel * (srcSize.width - kernelSize);
            Rpp32u increment = channel * (kernelSize - 1);
            for (int i = (2 * bound); i < srcSize.height; i++)
            {
                for (int j = (2 * bound); j < srcSize.width; j++)
                {
                    if (*dstPtrGreyscale32uTemp != 0)
                    {
                        if (channel == 3)
                        {
                            harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_minimum_kernel_host(dstPtrWindow + 1, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                            harris_corner_set_maximum_kernel_host(dstPtrWindow + 2, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                        else if (channel == 1)
                        {
                            harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                                chnFormat, channel);
                        }
                    }
                    dstPtrGreyscale32uTemp++;
                    dstPtrWindow += channel;
                }
                dstPtrGreyscale32uTemp += (kernelSize - 1);
                dstPtrWindow += increment;
            }
        }

        compute_padded_from_unpadded_host(dstPtr, srcSize, batch_srcSizeMax[batchCount], batch_dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtr);
        free(dstPtr);
        free(srcPtrGreyscale);
        free(srcPtrMod);
        free(dstPtrGreyscale);
        free(bresenhamCirclePositions);
        free(dstPtrGreyscale32u);
        free(dstPtrGreyscale32uMod);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus fast_corner_detector_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                      Rpp32u numOfPixels, T threshold,
                                      Rpp32u nonmaxKernelSize,
                                      RppiChnFormat chnFormat, Rpp32u channel)
{
    // RGB to Greyscale Conversion

    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u twiceImageDim = 2 * imageDim;

    T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
    T *srcPtrGreyscaleTemp;
    srcPtrGreyscaleTemp = srcPtrGreyscale;

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + imageDim;
            srcPtrTempB = srcPtrTempG + imageDim;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + 1;
            srcPtrTempB = srcPtrTempG + 1;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR += channel;
                srcPtrTempG += channel;
                srcPtrTempB += channel;
            }
        }
    }
    else if (channel == 1)
    {
        memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
    }

    Rpp32u newChannel = 1;

    // Pad image

    int bound = 3;
    RppiSize srcSizeMod;
    srcSizeMod.height = srcSize.height + (2 * bound);
    srcSizeMod.width = srcSize.width + (2 * bound);

    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * newChannel, sizeof(T));
    generate_evenly_padded_image_host(srcPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, RPPI_CHN_PLANAR, newChannel);

    // Compute the fast corner strengh matrix

    Rpp32u kernelSize = 7;
    T *dstPtrGreyscale = (T *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(T));
    T *dstPtrGreyscaleTemp;
    dstPtrGreyscaleTemp = dstPtrGreyscale;

    T *srcPtrWindow;
    srcPtrWindow = srcPtrMod;

    Rpp32u *bresenhamCirclePositions = (Rpp32u*) calloc(16, sizeof(Rpp32u));
    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    *bresenhamCirclePositionsTemp = 3;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = 4;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (1 * srcSizeMod.width) + 5;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (2 * srcSizeMod.width) + 6;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (3 * srcSizeMod.width) + 6;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (4 * srcSizeMod.width) + 6;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (5 * srcSizeMod.width) + 5;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 4;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 3;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (6 * srcSizeMod.width) + 2;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (5 * srcSizeMod.width) + 1;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (4 * srcSizeMod.width);
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (3 * srcSizeMod.width);
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (2 * srcSizeMod.width);
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = (1 * srcSizeMod.width) + 1;
    bresenhamCirclePositionsTemp++;
    *bresenhamCirclePositionsTemp = 2;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            fast_corner_detector_kernel_host(srcPtrWindow, dstPtrGreyscaleTemp, srcSizeMod,
                                             bresenhamCirclePositions, threshold, numOfPixels);
            srcPtrWindow++;
            dstPtrGreyscaleTemp++;
        }
        srcPtrWindow += (kernelSize - 1);
    }

    // Create score function

    generate_evenly_padded_image_host(dstPtrGreyscale, srcSize, srcPtrMod, srcSizeMod, RPPI_CHN_PLANAR, newChannel);

    Rpp32u *dstPtrGreyscale32u = (Rpp32u *)calloc(srcSize.height * srcSize.width * newChannel, sizeof(Rpp32u));
    Rpp32u *dstPtrGreyscale32uTemp;
    Rpp32u windowCenter;

    srcPtrWindow = srcPtrMod;
    dstPtrGreyscale32uTemp = dstPtrGreyscale32u;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            windowCenter = (Rpp32u) *(srcPtrWindow + (3 * srcSizeMod.width) + 3);
            if (windowCenter != 0)
            {
                fast_corner_detector_score_function_kernel_host(srcPtrWindow, dstPtrGreyscale32uTemp, srcSize, bresenhamCirclePositions, windowCenter);
            }
            else
            {
                *dstPtrGreyscale32uTemp = (Rpp32u) 0;
            }
            srcPtrWindow++;
            dstPtrGreyscale32uTemp++;
        }
        srcPtrWindow += (kernelSize - 1);
    }

    // Apply non max suppression

    int nonmaxBound = (nonmaxKernelSize - 1) / 2;
    RppiSize srcSizeNonmaxMod;
    srcSizeNonmaxMod.height = srcSize.height + (2 * nonmaxBound);
    srcSizeNonmaxMod.width = srcSize.width + (2 * nonmaxBound);

    Rpp32u *dstPtrGreyscale32uMod = (Rpp32u *)calloc(srcSizeNonmaxMod.height * srcSizeNonmaxMod.width * newChannel, sizeof(Rpp32u));
    generate_evenly_padded_image_host(dstPtrGreyscale32u, srcSize, dstPtrGreyscale32uMod, srcSizeNonmaxMod, RPPI_CHN_PLANAR, newChannel);

    Rpp32u *dstPtrGreyscale32uWindow;
    dstPtrGreyscale32uWindow = dstPtrGreyscale32uMod;
    dstPtrGreyscale32uTemp = dstPtrGreyscale32u;

    Rpp32u windowCenterPosIncrement = (nonmaxBound * srcSizeNonmaxMod.width) + nonmaxBound;
    Rpp32u remainingElementsInRow = srcSizeNonmaxMod.width - nonmaxKernelSize;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            windowCenter = (Rpp32u) *(dstPtrGreyscale32uWindow + windowCenterPosIncrement);
            non_max_suppression_kernel_host(dstPtrGreyscale32uWindow, dstPtrGreyscale32uTemp, srcSize,
                                nonmaxKernelSize, remainingElementsInRow, windowCenter,
                                RPPI_CHN_PLANAR, newChannel);
            dstPtrGreyscale32uWindow++;
            dstPtrGreyscale32uTemp++;
        }
        dstPtrGreyscale32uWindow += (nonmaxKernelSize - 1);
    }

    // Overlay Fast Corners on original image - Large Dot

    memcpy(dstPtr, srcPtr, channel * imageDim * sizeof(T));

    T *dstPtrWindow;
    kernelSize = 3;
    bound = 1;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        dstPtrGreyscale32uTemp = dstPtrGreyscale32u + (bound * srcSize.width) + bound;
        dstPtrWindow = dstPtr;
        Rpp32u remainingElementsInRow = srcSize.width - kernelSize;
        for (int i = (2 * bound); i < srcSize.height; i++)
        {
            for (int j = (2 * bound); j < srcSize.width; j++)
            {
                if (*dstPtrGreyscale32uTemp != 0)
                {
                    if (channel == 3)
                    {
                        harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_minimum_kernel_host(dstPtrWindow + imageDim, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_maximum_kernel_host(dstPtrWindow + twiceImageDim, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                    else if (channel == 1)
                    {
                        harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                }
                dstPtrGreyscale32uTemp++;
                dstPtrWindow++;
            }
            dstPtrGreyscale32uTemp += (kernelSize - 1);
            dstPtrWindow += (kernelSize - 1);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        dstPtrGreyscale32uTemp = dstPtrGreyscale32u + (channel * ((bound * srcSize.width) + bound));
        dstPtrWindow = dstPtr;
        Rpp32u remainingElementsInRow = channel * (srcSize.width - kernelSize);
        Rpp32u increment = channel * (kernelSize - 1);
        for (int i = (2 * bound); i < srcSize.height; i++)
        {
            for (int j = (2 * bound); j < srcSize.width; j++)
            {
                if (*dstPtrGreyscale32uTemp != 0)
                {
                    if (channel == 3)
                    {
                        harris_corner_set_minimum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_minimum_kernel_host(dstPtrWindow + 1, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                        harris_corner_set_maximum_kernel_host(dstPtrWindow + 2, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                    else if (channel == 1)
                    {
                        harris_corner_set_maximum_kernel_host(dstPtrWindow, kernelSize, remainingElementsInRow,
                                                              chnFormat, channel);
                    }
                }
                dstPtrGreyscale32uTemp++;
                dstPtrWindow += channel;
            }
            dstPtrGreyscale32uTemp += (kernelSize - 1);
            dstPtrWindow += increment;
        }
    }

    free(srcPtrGreyscale);
    free(srcPtrMod);
    free(dstPtrGreyscale);
    free(bresenhamCirclePositions);
    free(dstPtrGreyscale32u);
    free(dstPtrGreyscale32uMod);

    return RPP_SUCCESS;
}

/**************** Tensor Convert Bit Depth ***************/

template <typename T, typename U>
RppStatus tensor_convert_bit_depth_host(T* srcPtr, U* dstPtr,
                                        Rpp32u conversionType,
                                        Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    Rpp32u *tensorDimensionValuesTemp;
    tensorDimensionValuesTemp = tensorDimensionValues;

    Rpp32u tensorSize = 1;
    for(int i = 0; i < tensorDimension; i++)
    {
        tensorSize *= *tensorDimensionValuesTemp;
        tensorDimensionValuesTemp++;
    }

    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (conversionType == 1)
    {
        Rpp32s val = 128;
        for (int i = 0; i < tensorSize; i++)
        {
            *dstPtrTemp = (U) ((Rpp32s) *srcPtrTemp - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 2)
    {
        Rpp32f multiplier = 65535/255;
        for (int i = 0; i < tensorSize; i++)
        {
            *dstPtrTemp = (U) ((Rpp32f) *srcPtrTemp * multiplier);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (conversionType == 3)
    {
        Rpp32f multiplier = 65535/255;
        Rpp32f val = 32768;
        for (int i = 0; i < tensorSize; i++)
        {
            *dstPtrTemp = (U) (((Rpp32f) *srcPtrTemp * multiplier) - val);
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;

}

/**************** Tensor Transpose ***************/

template <typename T>
RppStatus tensor_transpose_host(T* srcPtr, T* dstPtr, Rpp32u *shape, Rpp32u *perm)
{
    T *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u numElements[4] = {
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1
    };

    for (int i = 0; i < shape[perm[0]]; i++)
    {
        for (int j = 0; j < shape[perm[1]]; j++)
        {
            for (int k = 0; k < shape[perm[2]]; k++)
            {
                for (int l = 0; l < shape[perm[3]]; l++)
                {
                    *dstPtrTemp = *(srcPtr + (
                        (i * numElements[perm[0]]) +
                        (j * numElements[perm[1]]) +
                        (k * numElements[perm[2]]) +
                        (l * numElements[perm[3]])
                    ));
                    dstPtrTemp++;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

/**************** hog ***************/

template <typename T, typename U>
RppStatus hog_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* batch_binsTensor, Rpp32u *batch_binsTensorLength,
                         RppiSize *batch_kernelSize, RppiSize *batch_windowSize,  Rpp32u *batch_windowStride, Rpp32u *batch_numOfBins,
                         Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u binsTensorLength = batch_binsTensorLength[batchCount];
        RppiSize kernelSize = batch_kernelSize[batchCount];
        RppiSize windowSize = batch_windowSize[batchCount];
        Rpp32u windowStride = batch_windowStride[batchCount];
        Rpp32u numOfBins = batch_numOfBins[batchCount];

        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);

        Rpp32u locHist = 0;
        compute_histogram_location_host(batch_binsTensorLength, batchCount, &locHist);
        U *binsTensor;
        binsTensor = batch_binsTensor + locHist;

        hog_host(srcPtr, batch_srcSize[batchCount], binsTensor, binsTensorLength,
                kernelSize, windowSize, windowStride, numOfBins, chnFormat, channel);
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus hog_host(T* srcPtr, RppiSize srcSize, U* binsTensor, Rpp32u binsTensorLength,
                   RppiSize kernelSize, RppiSize windowSize,  Rpp32u windowStride, Rpp32u numOfBins,
                   RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u newChannel = 1;

    Rpp32f gradientKernel[3] = {-1, 0, 1};
    RppiSize rppiGradientKernelSizeX, rppiGradientKernelSizeY;
    rppiGradientKernelSizeX.height = 1;
    rppiGradientKernelSizeX.width = 3;
    rppiGradientKernelSizeY.height = 3;
    rppiGradientKernelSizeY.width = 1;

    Rpp32s *gradientX = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    Rpp32s *gradientY = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    Rpp32s *gradientMagnitude = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    Rpp32f *gradientDirection = (Rpp32f *)calloc(imageDim * newChannel, sizeof(Rpp32f));

    T *srcPtrGreyscale = (T *)calloc(imageDim, sizeof(T));
    T *srcPtrGreyscaleTemp;
    srcPtrGreyscaleTemp = srcPtrGreyscale;

    if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + imageDim;
            srcPtrTempB = srcPtrTempG + imageDim;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            srcPtrTempR = srcPtr;
            srcPtrTempG = srcPtr + 1;
            srcPtrTempB = srcPtrTempG + 1;

            for (int i = 0; i < imageDim; i++)
            {
                *srcPtrGreyscaleTemp = (T) (((Rpp32u)(*srcPtrTempR) + (Rpp32u)(*srcPtrTempG) + (Rpp32u)(*srcPtrTempB)) / 3);
                srcPtrGreyscaleTemp++;
                srcPtrTempR += channel;
                srcPtrTempG += channel;
                srcPtrTempB += channel;
            }
        }
    }
    else if (channel == 1)
    {
        memcpy(srcPtrGreyscale, srcPtr, imageDim * sizeof(T));
    }

    hog_single_channel_gradient_computations_kernel_host(srcPtrGreyscale, srcSize, gradientX, gradientY, gradientMagnitude, gradientDirection,
                                                            gradientKernel, rppiGradientKernelSizeX, rppiGradientKernelSizeY);

    // if (channel == 3)
    // {
    //     Rpp32s *gradientX0 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    //     Rpp32s *gradientX1 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    //     Rpp32s *gradientX2 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    //     Rpp32s *gradientY0 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    //     Rpp32s *gradientY1 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));
    //     Rpp32s *gradientY2 = (Rpp32s *)calloc(imageDim * newChannel, sizeof(Rpp32s));

    //     T *srcPtrSingleChannel = (T *)calloc(imageDim * newChannel, sizeof(T));

    //     hog_three_channel_gradient_computations_kernel_host(srcPtr, srcPtrSingleChannel, srcSize,
    //                                                          gradientX0, gradientY0, gradientX1, gradientY1, gradientX2, gradientY2,
    //                                                          gradientX, gradientY,
    //                                                          gradientMagnitude, gradientDirection,
    //                                                          gradientKernel, rppiGradientKernelSizeX, rppiGradientKernelSizeY, chnFormat, channel);

    //     free(gradientX0);
    //     free(gradientX1);
    //     free(gradientX2);
    //     free(gradientY0);
    //     free(gradientY1);
    //     free(gradientY2);
    //     free(srcPtrSingleChannel);
    // }

    Rpp32s *gradientMagnitudeTemp, *gradientMagnitudeTemp2, *gradientMagnitudeTemp3;
    Rpp32f *gradientDirectionTemp, *gradientDirectionTemp2, *gradientDirectionTemp3;

    Rpp32u binsTensorTrueLength = 0;
    Rpp32u windowKernelHeightRatio = windowSize.height / kernelSize.height;
    Rpp32u windowKernelWidthRatio = windowSize.width / kernelSize.width;
    binsTensorTrueLength = ((windowKernelWidthRatio * windowKernelHeightRatio) + ((windowKernelWidthRatio - 1) * (windowKernelHeightRatio - 1)));
    Rpp32u numOfPositionsAlongImageWidth = (srcSize.width / windowStride - (windowSize.width / windowStride - 1));
    Rpp32u numOfPositionsAlongImageHeight = (srcSize.height / windowStride - (windowSize.height / windowStride - 1));
    binsTensorTrueLength = binsTensorTrueLength * (numOfPositionsAlongImageWidth * numOfPositionsAlongImageHeight);
    binsTensorTrueLength = binsTensorTrueLength * numOfBins;

    Rpp32u numOfPositionsAlongWindowWidth = (windowSize.width / kernelSize.width);
    Rpp32u numOfPositionsAlongWindowHeight = (windowSize.height / kernelSize.height);

    U *binsTensorTemp;
    binsTensorTemp = binsTensor;
    Rpp32f rangeInBin = PI / numOfBins;
    Rpp32u elementCount = 0, bin;
    Rpp32f adder = PI / 2;

    // For sliding window on image

    for(int i = 0; i < numOfPositionsAlongImageHeight; i++)
    {
        gradientMagnitudeTemp = gradientMagnitude + (i * windowStride * srcSize.width);
        gradientDirectionTemp = gradientDirection + (i * windowStride * srcSize.width);
        for(int j = 0; j < numOfPositionsAlongImageWidth; j++)
        {
            // For each window

            // Layer 1
            for (int m = 0; m < numOfPositionsAlongWindowHeight; m++)
            {
                gradientMagnitudeTemp2 = gradientMagnitudeTemp + (m * kernelSize.height * srcSize.width);
                gradientDirectionTemp2 = gradientDirectionTemp + (m * kernelSize.height * srcSize.width);
                for (int n = 0; n < numOfPositionsAlongWindowWidth; n++)
                {
                    U *kernelHistogram = (U*) calloc(numOfBins, sizeof(U));

                    // For each kernel
                    for (int p = 0; p < kernelSize.height; p++)
                    {
                        gradientMagnitudeTemp3 = gradientMagnitudeTemp2 + (p * srcSize.width);
                        gradientDirectionTemp3 = gradientDirectionTemp2 + (p * srcSize.width);
                        for (int q = 0; q < kernelSize.width; q++)
                        {
                            bin = (Rpp32u) ((*gradientDirectionTemp3 + adder) / rangeInBin);
                            if (bin > (numOfBins - 1))
                            {
                                bin = numOfBins - 1;
                            }
                            *(kernelHistogram + bin) += *gradientMagnitudeTemp3;
                            gradientMagnitudeTemp3++;
                            gradientDirectionTemp3++;
                        }
                    }
                    U *kernelHistogramTemp;
                    kernelHistogramTemp = kernelHistogram;
                    for (int r = 0; r < numOfBins; r++)
                    {
                        if (elementCount < (binsTensorLength - 1))
                        {
                            *binsTensorTemp = *kernelHistogramTemp;
                            binsTensorTemp++;
                            kernelHistogramTemp++;
                            elementCount++;
                        }
                        else
                        {
                            return RPP_SUCCESS;
                        }

                    }
                    gradientMagnitudeTemp2 += kernelSize.width;
                    gradientDirectionTemp2 += kernelSize.width;

                    free(kernelHistogram);
                }
            }

            // Layer 2
            for (int m = 0; m < numOfPositionsAlongWindowHeight - 1; m++)
            {
                gradientMagnitudeTemp2 = gradientMagnitudeTemp + (kernelSize.height / 2 * srcSize.width) + (m * kernelSize.height * srcSize.width) + (kernelSize.width / 2);
                gradientDirectionTemp2 = gradientDirectionTemp + (kernelSize.height / 2 * srcSize.width)+ (m * kernelSize.height * srcSize.width) + (kernelSize.width / 2);
                for (int n = 0; n < numOfPositionsAlongWindowWidth - 1; n++)
                {
                    U *kernelHistogram = (U*) calloc(numOfBins, sizeof(U));

                    // For each kernel

                    for (int p = 0; p < kernelSize.height; p++)
                    {
                        gradientMagnitudeTemp3 = gradientMagnitudeTemp2 + (p * srcSize.width);
                        gradientDirectionTemp3 = gradientDirectionTemp2 + (p * srcSize.width);
                        for (int q = 0; q < kernelSize.width; q++)
                        {
                            bin = (Rpp32u) ((*gradientDirectionTemp3 + adder) / rangeInBin);
                            if (bin > (numOfBins - 1))
                            {
                                bin = numOfBins - 1;
                            }
                            *(kernelHistogram + bin) += *gradientMagnitudeTemp3;
                            gradientMagnitudeTemp3++;
                            gradientDirectionTemp3++;
                        }
                    }
                    U *kernelHistogramTemp;
                    kernelHistogramTemp = kernelHistogram;
                    for (int r = 0; r < numOfBins; r++)
                    {
                        if (elementCount < (binsTensorLength - 1))
                        {
                            *binsTensorTemp = *kernelHistogramTemp;
                            binsTensorTemp++;
                            kernelHistogramTemp++;
                            elementCount++;
                        }
                        else
                        {
                            return RPP_SUCCESS;
                        }

                    }

                    gradientMagnitudeTemp2 += kernelSize.width;
                    gradientDirectionTemp2 += kernelSize.width;

                    free(kernelHistogram);
                }
            }

            gradientMagnitudeTemp += windowStride;
            gradientDirectionTemp += windowStride;
        }
    }

    free(gradientX);
    free(gradientY);
    free(gradientMagnitude);
    free(gradientDirection);

    return RPP_SUCCESS;
}

/**************** Tensor Matrix Multiply ***************/

template <typename T>
RppStatus tensor_matrix_multiply_host(T* srcPtr1, T* srcPtr2, T* dstPtr,
                          Rpp32u *tensorDimensionValues1, Rpp32u *tensorDimensionValues2)
{
    if (*(tensorDimensionValues1 + 1) != *tensorDimensionValues2)
    {
        return RPP_ERROR;
    }

    T *srcPtr1Temp;

    Rpp32u outputCols = *(tensorDimensionValues2 + 1);
    Rpp32u pixel;

    srcPtr1Temp = srcPtr1;
    for (int i = 0; i < *tensorDimensionValues1; i++)
    {
        T *dstPtrRow;
        dstPtrRow = dstPtr + (i * outputCols);
        T *srcPtr2Temp;
        srcPtr2Temp = srcPtr2;
        for (int k = 0; k < *tensorDimensionValues2; k++)
        {
            T *dstPtrCol;
            dstPtrCol = dstPtrRow;
            for (int j = 0; j < outputCols; j++)
            {
                pixel = (Rpp32u) *dstPtrCol + ((Rpp32u) *srcPtr1Temp * (Rpp32u) *srcPtr2Temp);
                pixel = RPPPIXELCHECK(pixel);
                *dstPtrCol = (T) pixel;
                dstPtrCol++;
                srcPtr2Temp++;
            }
            srcPtr1Temp++;
        }
    }

    return RPP_SUCCESS;

}

#endif
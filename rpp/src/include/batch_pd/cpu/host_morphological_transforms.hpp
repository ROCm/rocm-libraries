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

#ifndef HOST_MORPHOLOGICAL_TRANSFORMS_HPP
#define HOST_MORPHOLOGICAL_TRANSFORMS_HPP

#include "rpp_cpu_batchpd.hpp"

/**************** erode ***************/

template <typename T>
RppStatus erode_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                           Rpp32u *batch_kernelSize,
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

            Rpp32f kernelSize = batch_kernelSize[batchCount];
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
            Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowBoundedROI = srcSizeBoundedROI.width;
            Rpp32u elementsInRowROI = srcSizeROI.width;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= bound) &&(y1 >= bound))
            {
                T *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;
                for (int c = 0; c < channel; c++)
                {
                    T *srcPtrImageTemp;
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
                T *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;

                srcPtrBoundedROITemp += bound;
                for (int c = 0; c < channel; c++)
                {
                    T *srcPtrImageTemp;
                    srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + ((Rpp32u) y1 * elementsInRowMax) + (Rpp32u) x1;

                    for (int i = 0; i < bound; i++)
                    {
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                        srcPtrBoundedROITemp += elementsInRowBoundedROI;
                    }
                    srcPtrBoundedROITemp -= bound;

                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        for (int i = 0; i < bound; i++)
                        {
                            memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, 1 * sizeof(T));
                            srcPtrBoundedROITemp += 1;
                        }
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                        srcPtrBoundedROITemp += elementsInRowROI;
                        for (int i = 0; i < bound; i++)
                        {
                            memcpy(srcPtrBoundedROITemp, srcPtrImageTemp + elementsInRowROI, 1 * sizeof(T));
                            srcPtrBoundedROITemp += 1;
                        }

                        srcPtrImageTemp += elementsInRowMax;
                    }
                    srcPtrImageTemp -= elementsInRowMax;

                    srcPtrBoundedROITemp += bound;
                    for (int i = 0; i < bound; i++)
                    {
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                        srcPtrBoundedROITemp += elementsInRowBoundedROI;
                    }
                }
            }

            Rpp32u remainingElementsInRow = srcSizeBoundedROI.width - rppiKernelSize.width;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32u roiRowCount = 0;

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
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
                        T *srcPtrWindow;
                        srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                erode_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                  kernelSize, remainingElementsInRow,
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

            Rpp32f kernelSize = batch_kernelSize[batchCount];
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
                T *srcPtrImageTemp, *srcPtrBoundedROITemp;
                srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
                srcPtrBoundedROITemp = srcPtrBoundedROI;

                srcPtrBoundedROITemp += (bound * channel);

                for (int i = 0; i < bound; i++)
                {
                    memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                    srcPtrBoundedROITemp += elementsInRowBoundedROI;
                }
                srcPtrBoundedROITemp -= (bound * channel);

                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    for (int i = 0; i < bound; i++)
                    {
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, channel * sizeof(T));
                        srcPtrBoundedROITemp += channel;
                    }
                    memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                    srcPtrBoundedROITemp += elementsInRowROI;
                    for (int i = 0; i < bound; i++)
                    {
                        memcpy(srcPtrBoundedROITemp, srcPtrImageTemp + elementsInRowROI, channel * sizeof(T));
                        srcPtrBoundedROITemp += channel;
                    }

                    srcPtrImageTemp += elementsInRowMax;
                }
                srcPtrImageTemp -= elementsInRowMax;

                srcPtrBoundedROITemp += (bound * channel);
                for (int i = 0; i < bound; i++)
                {
                    memcpy(srcPtrBoundedROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                    srcPtrBoundedROITemp += elementsInRowBoundedROI;
                }
            }

            Rpp32u remainingElementsInRow = (srcSizeBoundedROI.width - rppiKernelSize.width) * channel;

            Rpp32u roiRowCount = 0;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
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
                    T *srcPtrWindow;
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
                                erode_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                  kernelSize, remainingElementsInRow,
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
RppStatus erode_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;
        Rpp32u rowIncrement = kernelSize - 1;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    erode_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += rowIncrement;
            }
            srcPtrWindow += (rowIncrement * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize) * channel;
        Rpp32u rowIncrement = (kernelSize - 1) * channel;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    erode_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += rowIncrement;
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** dilate ***************/

template <typename T>
RppStatus dilate_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                           Rpp32u *batch_kernelSize,
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

            Rpp32f kernelSize = batch_kernelSize[batchCount];
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
            Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= bound) &&(y1 >= bound))
            {
                T *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;
                for (int c = 0; c < channel; c++)
                {
                    T *srcPtrImageTemp;
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

                T *srcPtrROITemp;
                srcPtrROITemp = srcPtrROI;
                for (int c = 0; c < channel; c++)
                {
                    T *srcPtrImageTemp;
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

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32u roiRowCount = 0;


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
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
                        T *srcPtrWindow;
                        srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                dilate_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                  kernelSize, remainingElementsInRow,
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

            Rpp32f kernelSize = batch_kernelSize[batchCount];
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

            Rpp32u roiRowCount = 0;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
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
                    T *srcPtrWindow;
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
                                dilate_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                  kernelSize, remainingElementsInRow,
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
RppStatus dilate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;
        Rpp32u rowIncrement = kernelSize - 1;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    dilate_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += rowIncrement;
            }
            srcPtrWindow += (rowIncrement * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize) * channel;
        Rpp32u rowIncrement = (kernelSize - 1) * channel;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    dilate_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += rowIncrement;
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

#endif // #ifndef HOST_MORPHOLOGICAL_TRANSFORMS_HPP

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

#ifndef HOST_FILTER_OPERATIONS_HPP
#define HOST_FILTER_OPERATIONS_HPP

#include "rpp_cpu_batchpd.hpp"

/**************** box_filter ***************/

template <typename T>
RppStatus box_filter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
            Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_box_kernel_host(kernel, kernelSize);
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

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(kernel);
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
            Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_box_kernel_host(kernel, kernelSize);
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

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(kernel);
            free(srcPtrBoundedROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus box_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_box_kernel_host(kernel, kernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    free(kernel);
    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** median_filter ***************/

template <typename T>
RppStatus median_filter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
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
                                median_filter_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
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
                                median_filter_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
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
RppStatus median_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRowPlanar,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRowPacked,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** gaussian_filter ***************/

template <typename T>
RppStatus gaussian_filter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                     Rpp32f *batch_stdDev, Rpp32u *batch_kernelSize,
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

            Rpp32f stdDev = batch_stdDev[batchCount];

            Rpp32f kernelSize = batch_kernelSize[batchCount];
            Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_gaussian_kernel_host(stdDev, kernel, kernelSize);
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

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(kernel);
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

            Rpp32f stdDev = batch_stdDev[batchCount];

            Rpp32f kernelSize = batch_kernelSize[batchCount];
            Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_gaussian_kernel_host(stdDev, kernel, kernelSize);
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

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(kernel);
            free(srcPtrBoundedROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus gaussian_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    free(kernel);
    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** nonlinear_filter ***************/

template <typename T>
RppStatus nonlinear_filter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
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
                                median_filter_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
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
                                median_filter_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
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
RppStatus nonlinear_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRowPlanar,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRowPacked,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** non_max_suppression ***************/

template <typename T>
RppStatus non_max_suppression_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
                srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDim);
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32u roiRowCount = 0;
                Rpp32u windowCenterPosIncrement = (bound * srcSizeBoundedROI.width) + bound;


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
                            T windowCenter = (T) *(srcPtrWindow + windowCenterPosIncrement);
                            if((x1 <= j) && (j <= x2 ))
                            {
                                non_max_suppression_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                                kernelSize, remainingElementsInRow, windowCenter,
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
            Rpp32u windowCenterPosIncrement = channel * ((bound * srcSizeBoundedROI.width) + bound);


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
                                T windowCenter = (T) *(srcPtrWindow + windowCenterPosIncrement);
                                non_max_suppression_kernel_host(srcPtrWindow, dstPtrTemp, batch_srcSize[batchCount],
                                                                kernelSize, remainingElementsInRow, windowCenter,
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
RppStatus non_max_suppression_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    T *srcPtrWindow, *dstPtrTemp;
    T windowCenter;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u windowCenterPosIncrement = (bound * srcSizeMod.width) + bound;
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    windowCenter = (T) *(srcPtrWindow + windowCenterPosIncrement);
                    non_max_suppression_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow, windowCenter,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u windowCenterPosIncrement = channel * ((bound * srcSizeMod.width) + bound);
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize) * channel;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    windowCenter = (T) *(srcPtrWindow + windowCenterPosIncrement);
                    non_max_suppression_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                      kernelSize, remainingElementsInRow, windowCenter,
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** sobel_filter ***************/

template <typename T>
RppStatus sobel_filter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                  Rpp32u *batch_sobelType,
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

            Rpp32u sobelType = batch_sobelType[batchCount];

            Rpp32f kernelSize = 3;
            Rpp32f *kernelX = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            Rpp32f *kernelY = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_sobel_kernel_host(kernelX, 1);
            generate_sobel_kernel_host(kernelY, 2);
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

            if (sobelType == 0)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelX, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
            }
            else if (sobelType == 1)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelY, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
            }
            else if (sobelType == 2)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelX, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
                T *image1PtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], image1PtrUnpadded,
                                          chnFormat, channel);
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelY, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
                T *image2PtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], image2PtrUnpadded,
                                          chnFormat, channel);
                T *imageOutPtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_magnitude_ROI_host(image1PtrUnpadded, image2PtrUnpadded, batch_srcSize[batchCount], imageOutPtrUnpadded, x1, y1, x2, y2, chnFormat, channel);
                compute_padded_from_unpadded_host(imageOutPtrUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                          chnFormat, channel);

                free(image1PtrUnpadded);
                free(image2PtrUnpadded);
                free(imageOutPtrUnpadded);
            }

            free(kernelX);
            free(kernelY);
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

            Rpp32u sobelType = batch_sobelType[batchCount];

            Rpp32f kernelSize = 3;
            Rpp32f *kernelX = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            Rpp32f *kernelY = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
            int bound = ((kernelSize - 1) / 2);
            generate_sobel_kernel_host(kernelX, 1);
            generate_sobel_kernel_host(kernelY, 2);
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

            if (sobelType == 0)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelX, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
            }
            else if (sobelType == 1)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelY, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
            }
            else if (sobelType == 2)
            {
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelX, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
                T *image1PtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], image1PtrUnpadded,
                                          chnFormat, channel);
                convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernelY, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);
                T *image2PtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], image2PtrUnpadded,
                                          chnFormat, channel);
                T *imageOutPtrUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                compute_magnitude_ROI_host(image1PtrUnpadded, image2PtrUnpadded, batch_srcSize[batchCount], imageOutPtrUnpadded, x1, y1, x2, y2, chnFormat, channel);
                compute_padded_from_unpadded_host(imageOutPtrUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                          chnFormat, channel);

                free(image1PtrUnpadded);
                free(image2PtrUnpadded);
                free(imageOutPtrUnpadded);
            }

            free(kernelX);
            free(kernelY);
            free(srcPtrBoundedROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus sobel_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                            Rpp32u sobelType,
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + 2;
    srcSizeMod.height = srcSize.height + 2;

    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));
    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = 3;
    rppiKernelSize.width = 3;

    if (sobelType == 0)
    {
        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernelX, rppiKernelSize, chnFormat, channel);

        free(kernelX);
    }
    else if (sobelType == 1)
    {
        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernelY, rppiKernelSize, chnFormat, channel);

        free(kernelY);
    }
    else if (sobelType == 2)
    {
        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        Rpp32s *dstPtrIntermediateX = (Rpp32s *)calloc(srcSize.height * srcSize.width * channel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateX, srcSize, kernelX, rppiKernelSize, chnFormat, channel);

        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        Rpp32s *dstPtrIntermediateY = (Rpp32s *)calloc(srcSize.height * srcSize.width * channel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateY, srcSize, kernelY, rppiKernelSize, chnFormat, channel);

        compute_magnitude_host(dstPtrIntermediateX, dstPtrIntermediateY, srcSize, dstPtr, chnFormat, channel);

        free(kernelX);
        free(kernelY);
        free(dstPtrIntermediateX);
        free(dstPtrIntermediateY);
    }

    free(srcPtrMod);

    return RPP_SUCCESS;
}

/**************** custom_convolution ***************/

template <typename T>
RppStatus custom_convolution_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                        Rpp32f *batch_kernel, RppiSize *batch_rppiKernelSize,
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

            RppiSize rppiKernelSize;
            rppiKernelSize.height = batch_rppiKernelSize[batchCount].height;
            rppiKernelSize.width = batch_rppiKernelSize[batchCount].width;

            Rpp32u numOfElements = rppiKernelSize.height * rppiKernelSize.width;
            Rpp32f *kernel = (Rpp32f*) calloc(numOfElements, sizeof(Rpp32f));
            for (int i = 0; i < numOfElements; i++)
            {
                kernel[i] = batch_kernel[(batchCount * numOfElements) + i];
            }

            int boundY = ((rppiKernelSize.height - 1) / 2);
            int boundX = ((rppiKernelSize.width - 1) / 2);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            RppiSize srcSizeBoundedROI;
            srcSizeBoundedROI.height = roiPoints[batchCount].roiHeight + (2 * boundY);
            srcSizeBoundedROI.width = roiPoints[batchCount].roiWidth + (2 * boundX);
            Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= boundY) &&(y1 >= boundX))
            {
                T *srcPtrImageTemp, *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;
                for (int c = 0; c < channel; c++)
                {
                    srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + (((Rpp32u) y1 - boundY) * batch_srcSizeMax[batchCount].width) + ((Rpp32u) x1 - boundX);
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

                RppiSize srcSizeCornerBoundedROI;
                srcSizeCornerBoundedROI.height = roiPoints[batchCount].roiHeight + boundY;
                srcSizeCornerBoundedROI.width = roiPoints[batchCount].roiWidth + boundX;
                T *srcPtrCornerBoundedROI = (T *)calloc(srcSizeCornerBoundedROI.height * srcSizeCornerBoundedROI.width * channel, sizeof(T));

                generate_corner_padded_image_host(srcPtrROI, srcSizeROI, srcPtrCornerBoundedROI, srcSizeCornerBoundedROI, 1, chnFormat, channel);

                generate_corner_padded_image_host(srcPtrCornerBoundedROI, srcSizeCornerBoundedROI, srcPtrBoundedROI, srcSizeBoundedROI, 4, chnFormat, channel);

                free(srcPtrROI);
                free(srcPtrCornerBoundedROI);
            }

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(srcPtrBoundedROI);
            free(kernel);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

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

            RppiSize rppiKernelSize;
            rppiKernelSize.height = batch_rppiKernelSize[batchCount].height;
            rppiKernelSize.width = batch_rppiKernelSize[batchCount].width;

            Rpp32u numOfElements = rppiKernelSize.height * rppiKernelSize.width;
            Rpp32f *kernel = (Rpp32f*) calloc(numOfElements, sizeof(Rpp32f));
            for (int i = 0; i < numOfElements; i++)
            {
                kernel[i] = batch_kernel[(batchCount * numOfElements) + i];
            }

            int boundY = ((rppiKernelSize.height - 1) / 2);
            int boundX = ((rppiKernelSize.width - 1) / 2);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            RppiSize srcSizeBoundedROI;
            srcSizeBoundedROI.height = roiPoints[batchCount].roiHeight + (2 * boundY);
            srcSizeBoundedROI.width = roiPoints[batchCount].roiWidth + (2 * boundX);
            T *srcPtrBoundedROI = (T *)calloc(srcSizeBoundedROI.height * srcSizeBoundedROI.width * channel, sizeof(T));

            RppiSize srcSizeROI;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowBoundedROI = channel * srcSizeBoundedROI.width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            if ((srcSizeBoundedROI.height <= batch_srcSize[batchCount].height) &&
            (srcSizeBoundedROI.width <= batch_srcSize[batchCount].width) &&(x1 >= boundY) &&(y1 >= boundX))
            {
                T *srcPtrImageTemp, *srcPtrBoundedROITemp;
                srcPtrBoundedROITemp = srcPtrBoundedROI;

                srcPtrImageTemp = srcPtrImage + (((Rpp32u) y1 - boundY) * elementsInRowMax) + (channel * ((Rpp32u) x1 - boundX));
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

                RppiSize srcSizeCornerBoundedROI;
                srcSizeCornerBoundedROI.height = roiPoints[batchCount].roiHeight + boundY;
                srcSizeCornerBoundedROI.width = roiPoints[batchCount].roiWidth + boundX;
                T *srcPtrCornerBoundedROI = (T *)calloc(srcSizeCornerBoundedROI.height * srcSizeCornerBoundedROI.width * channel, sizeof(T));

                generate_corner_padded_image_host(srcPtrROI, srcSizeROI, srcPtrCornerBoundedROI, srcSizeCornerBoundedROI, 1, chnFormat, channel);

                generate_corner_padded_image_host(srcPtrCornerBoundedROI, srcSizeCornerBoundedROI, srcPtrBoundedROI, srcSizeBoundedROI, 4, chnFormat, channel);

                free(srcPtrROI);
                free(srcPtrCornerBoundedROI);
            }

            convolve_image_host_batch(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage,
                                      srcPtrBoundedROI, srcSizeBoundedROI,
                                      kernel, rppiKernelSize,
                                      x1, y1, x2, y2,
                                      chnFormat, channel);

            free(srcPtrBoundedROI);
            free(kernel);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus custom_convolution_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                  Rpp32f *kernel, RppiSize rppiKernelSize,
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    custom_convolve_image_host(srcPtr, srcSize, dstPtr, kernel, rppiKernelSize, chnFormat, channel);

    return RPP_SUCCESS;
}

// /**************** Bilateral Filter ***************/

template <typename T>
RppStatus bilateral_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32u kernelSize, Rpp32f sigmaI, Rpp32f sigmaS,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.height = srcSize.height + (2 * bound);
    srcSizeMod.width = srcSize.width + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    Rpp32u incrementToWindowCenterPlanar = (bound * srcSizeMod.width) + bound;
    Rpp32u incrementToWindowCenterPacked = ((bound * srcSizeMod.width) + bound) * channel;
    Rpp32f multiplierI, multiplierS, multiplier;
    multiplierI = -1 / (2 * sigmaI * sigmaI);
    multiplierS = -1 / (2 * sigmaS * sigmaS);
    multiplier = 1 / (4 * M_PI * M_PI * sigmaI * sigmaI * sigmaS * sigmaS);

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    generate_bilateral_kernel_host<T>(multiplierI, multiplierS, multiplier, kernel, kernelSize, bound,
                                                      srcPtrWindow, srcSizeMod, remainingElementsInRowPlanar, incrementToWindowCenterPlanar,
                                                      chnFormat, channel);
                   // convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                              //              kernel, kernelSize, remainingElementsInRowPlanar,
                                    //        chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    generate_bilateral_kernel_host<T>(multiplierI, multiplierS, multiplier, kernel, kernelSize, bound,
                                                      srcPtrWindow, srcSizeMod, remainingElementsInRowPacked, incrementToWindowCenterPacked,
                                                      chnFormat, channel);
                    //convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                      //                      kernel, kernelSize, remainingElementsInRowPacked,
                        //                    chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }

    free(kernel);
    free(srcPtrMod);

    return RPP_SUCCESS;

}

#endif
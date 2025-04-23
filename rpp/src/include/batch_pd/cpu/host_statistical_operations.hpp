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

#ifndef STATISTICAL_OPERATIONS_HPP
#define STATISTICAL_OPERATIONS_HPP

#include <limits>
#include "rpp_cpu_batchpd.hpp"

/**************** min ***************/

template <typename T>
RppStatus min_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
                srcPtr1Channel = srcPtr1Image + (c * imageDimMax);
                srcPtr2Channel = srcPtr2Image + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Channel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtr2Temp = srcPtr2Channel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, batch_srcSize[batchCount].width * sizeof(T));

                        srcPtr1Temp += batch_srcSizeMax[batchCount].width;
                        srcPtr2Temp += batch_srcSizeMax[batchCount].width;
                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *dstPtrTemp = RPPMIN2(*srcPtr1Temp, *srcPtr2Temp);

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtr1Temp;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                        }
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

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                    srcPtr1Temp += elementsInRowMax;
                    srcPtr2Temp += elementsInRowMax;
                    dstPtrTemp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtr1Temp, channel * sizeof(T));

                            srcPtr1Temp += channel;
                            srcPtr2Temp += channel;
                            dstPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = RPPMIN2(*srcPtr1Temp, *srcPtr2Temp);

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus min_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                RppiChnFormat chnFormat,Rpp32u channel)
{
    compute_min_host(srcPtr1, srcPtr2, srcSize, dstPtr, channel);

    return RPP_SUCCESS;

}

/**************** max ***************/

template <typename T>
RppStatus max_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
                srcPtr1Channel = srcPtr1Image + (c * imageDimMax);
                srcPtr2Channel = srcPtr2Image + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Channel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtr2Temp = srcPtr2Channel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, batch_srcSize[batchCount].width * sizeof(T));

                        srcPtr1Temp += batch_srcSizeMax[batchCount].width;
                        srcPtr2Temp += batch_srcSizeMax[batchCount].width;
                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *dstPtrTemp = RPPMAX2(*srcPtr1Temp, *srcPtr2Temp);

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtr1Temp;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                        }
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

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                    srcPtr1Temp += elementsInRowMax;
                    srcPtr2Temp += elementsInRowMax;
                    dstPtrTemp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtr1Temp, channel * sizeof(T));

                            srcPtr1Temp += channel;
                            srcPtr2Temp += channel;
                            dstPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = RPPMAX2(*srcPtr1Temp, *srcPtr2Temp);

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus max_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                RppiChnFormat chnFormat,Rpp32u channel)
{
    compute_max_host(srcPtr1, srcPtr2, srcSize, dstPtr, channel);

    return RPP_SUCCESS;

}

/**************** thresholding ***************/

template <typename T>
RppStatus thresholding_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                  T *batch_min, T *batch_max,
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

            T min = batch_min[batchCount];
            T max = batch_max[batchCount];

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

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                if (*srcPtrTemp < min)
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (*srcPtrTemp <= max)
                                {
                                    *dstPtrTemp = (T) 255;
                                }
                                else
                                {
                                    *dstPtrTemp = (T) 0;
                                }

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

            T min = batch_min[batchCount];
            T max = batch_max[batchCount];

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

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
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
                                if (*srcPtrTemp < min)
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (*srcPtrTemp <= max)
                                {
                                    *dstPtrTemp = (T) 255;
                                }
                                else
                                {
                                    *dstPtrTemp = (T) 0;
                                }

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus thresholding_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                 U min, U max,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_threshold_host(srcPtr, srcSize, dstPtr, min, max, 1, chnFormat, channel);

    return RPP_SUCCESS;

}

/**************** histogram ***************/

template <typename T>
RppStatus histogram_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32u *outputHistogram,
                               Rpp32u bins,
                               Rpp32u nbatchSize,
                               RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            //Rpp32u bins = batch_bins[batchCount];
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage;
            //Rpp32u histogramLoc = 0;
            //compute_histogram_location_host(batch_bins, batchCount, &histogramLoc);
            //outputHistogramImage = outputHistogram + histogramLoc;
            outputHistogramImage = outputHistogram + (bins * batchCount);

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);

                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            // #pragma omp critical
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;

                            srcPtrTemp++;
                        }
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

            //Rpp32u bins = batch_bins[batchCount];
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage;
            //Rpp32u histogramLoc = 0;
            //compute_histogram_location_host(batch_bins, batchCount, &histogramLoc);
            //outputHistogramImage = outputHistogram + histogramLoc;
            outputHistogramImage = outputHistogram + (bins * batchCount);

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;

                            srcPtrTemp++;
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus histogram_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins,
                            Rpp32u channel)
{
    histogram_kernel_host(srcPtr, srcSize, outputHistogram, bins - 1, channel);

    return RPP_SUCCESS;

}

/**************** histogram_equalization ***************/

template <typename T>
RppStatus histogram_equalization_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                            Rpp32u nbatchSize,
                                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32u bins = 256;
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage = (Rpp32u*) calloc(bins, sizeof(Rpp32u));
            T *lookUpTable = (T *) calloc (bins, sizeof(T));
            Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * imageDim));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);

                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;

                            srcPtrTemp++;
                        }
                    }
                }
            }

            Rpp32u sum = 0;
            Rpp32u *outputHistogramImageTemp;
            T *lookUpTableTemp;
            outputHistogramImageTemp = outputHistogramImage;
            lookUpTableTemp = lookUpTable;

            for (int i = 0; i < 256; i++)
            {
                sum += *outputHistogramImageTemp;
                *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
                outputHistogramImageTemp++;
                lookUpTableTemp++;
            }

            Rpp32f x1 = 0;
            Rpp32f y1 = 0;
            Rpp32f x2 = 0;
            Rpp32f y2 = 0;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
            }

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

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtrTemp;
                            }
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }

            free(outputHistogramImage);
            free(lookUpTable);
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

            Rpp32u bins = 256;
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage = (Rpp32u*) calloc(bins, sizeof(Rpp32u));
            T *lookUpTable = (T *) calloc (bins, sizeof(T));
            Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * imageDim));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;

                            srcPtrTemp++;
                        }
                    }
                }
            }

            Rpp32u sum = 0;
            Rpp32u *outputHistogramImageTemp;
            T *lookUpTableTemp;
            outputHistogramImageTemp = outputHistogramImage;
            lookUpTableTemp = lookUpTable;

            for (int i = 0; i < 256; i++)
            {
                sum += *outputHistogramImageTemp;
                *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
                outputHistogramImageTemp++;
                lookUpTableTemp++;
            }

            Rpp32f x1 = 0;
            Rpp32f y1 = 0;
            Rpp32f x2 = 0;
            Rpp32f y2 = 0;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
            }



            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                }
                else
                {
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
                                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }

            free(outputHistogramImage);
            free(lookUpTable);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus histogram_equalization_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat,Rpp32u channel)
{
    Rpp32u histogram[256];
    T lookUpTable[256];
    Rpp32u *histogramTemp;
    T *lookUpTableTemp;
    Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * srcSize.height * srcSize.width));

    histogram_kernel_host(srcPtr, srcSize, histogram, 255, channel);

    Rpp32u sum = 0;
    histogramTemp = histogram;
    lookUpTableTemp = lookUpTable;

    for (int i = 0; i < 256; i++)
    {
        sum += *histogramTemp;
        *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
        histogramTemp++;
        lookUpTableTemp++;
    }

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
        srcPtrTemp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** min_max_loc ***************/

template <typename T>
RppStatus min_max_loc_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
                                 Rpp8u *batch_min, Rpp8u *batch_max, Rpp32u *batch_minLoc, Rpp32u *batch_maxLoc,
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

            Rpp8u *min = batch_min + batchCount;
            Rpp8u *max = batch_max + batchCount;
            Rpp32u *minLoc = batch_minLoc + batchCount;
            Rpp32u *maxLoc = batch_maxLoc + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            T *maxLocPtrTemp, *minLocPtrTemp;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        // #pragma omp critical
                        if (*srcPtrTemp > *max)
                        {
                            *max = *srcPtrTemp;
                            maxLocPtrTemp = srcPtrTemp;
                        }

                        // #pragma omp critical
                        if (*srcPtrTemp < *min)
                        {
                            *min = *srcPtrTemp;
                            minLocPtrTemp = srcPtrTemp;
                        }

                        srcPtrTemp++;
                    }
                }
            }
            *minLoc = minLocPtrTemp - srcPtrImage;
            *maxLoc = maxLocPtrTemp - srcPtrImage;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp8u *min = batch_min + batchCount;
            Rpp8u *max = batch_max + batchCount;
            Rpp32u *minLoc = batch_minLoc + batchCount;
            Rpp32u *maxLoc = batch_maxLoc + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            T *maxLocPtrTemp, *minLocPtrTemp;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        // #pragma omp critical
                        if (*srcPtrTemp > *max)
                        {
                            *max = *srcPtrTemp;
                            maxLocPtrTemp = srcPtrTemp;
                        }

                        // #pragma omp critical
                        if (*srcPtrTemp < *min)
                        {
                            *min = *srcPtrTemp;
                            minLocPtrTemp = srcPtrTemp;
                        }

                        srcPtrTemp++;
                    }
                }
            }
            *minLoc = minLocPtrTemp - srcPtrImage;
            *maxLoc = maxLocPtrTemp - srcPtrImage;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus min_max_loc_host(T* srcPtr, RppiSize srcSize,
                         Rpp8u* min, Rpp8u* max, Rpp32u* minLoc, Rpp32u* maxLoc,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    *min = 255;
    *max = 0;

    T *srcPtrTemp, *minLocPtrTemp, *maxLocPtrTemp;
    srcPtrTemp = srcPtr;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        if (*srcPtrTemp > *max)
        {
            *max = *srcPtrTemp;
            maxLocPtrTemp = srcPtrTemp;
        }
        if (*srcPtrTemp < *min)
        {
            *min = *srcPtrTemp;
            minLocPtrTemp = srcPtrTemp;
        }
        srcPtrTemp++;
    }
    *minLoc = minLocPtrTemp - srcPtr;
    *maxLoc = maxLocPtrTemp - srcPtr;

    return RPP_SUCCESS;

}

/**************** mean_stddev ***************/

// Without ROI
template <typename T>
RppStatus mean_stddev_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
                                 Rpp32f *batch_mean, Rpp32f *batch_stddev,
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

            Rpp32f *mean, *stddev;
            mean = batch_mean + batchCount;
            stddev = batch_stddev + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        // #pragma omp critical
                        *mean = *mean + (Rpp32f) *srcPtrTemp;
                        srcPtrTemp++;
                    }
                }
            }

            *mean = *mean / (channel * imageDim);

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        // #pragma omp critical
                        *stddev = *stddev + (((*mean) - (*srcPtrTemp)) * ((*mean) - (*srcPtrTemp)));
                        srcPtrTemp++;
                    }
                }
            }

            *stddev = sqrt((*stddev) / (imageDim * channel));

        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32f *mean, *stddev;
            mean = batch_mean + batchCount;
            stddev = batch_stddev + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        // #pragma omp critical
                        *mean = *mean + (Rpp32f) *srcPtrTemp;

                        srcPtrTemp++;
                    }
                }
            }

            *mean = *mean / (channel * imageDim);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        // #pragma omp critical
                        *stddev = *stddev + (((*mean) - (*srcPtrTemp)) * ((*mean) - (*srcPtrTemp)));

                        srcPtrTemp++;
                    }
                }
            }

            *stddev = sqrt((*stddev) / (imageDim * channel));
        }
    }

    return RPP_SUCCESS;
}

// With ROI
/*
template <typename T>
RppStatus mean_stddev_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
                                 Rpp32f *batch_mean, Rpp32f *batch_stddev,
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

            Rpp32f *mean, *stddev;
            mean = batch_mean + batchCount;
            stddev = batch_stddev + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);

                omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            #pragma omp critical
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *mean = *mean + (Rpp32f) *srcPtrTemp;

                                srcPtrTemp++;
                            }
                            else
                            {
                                srcPtrTemp++;
                            }
                        }
                    }
                }
            }

            *mean = *mean / (channel * imageDim);

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);

                omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            #pragma omp critical
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *stddev = *stddev + (((*mean) - (*srcPtrTemp)) * ((*mean) - (*srcPtrTemp)));

                                srcPtrTemp++;
                            }
                            else
                            {
                                srcPtrTemp++;
                            }
                        }
                    }
                }
            }

            *stddsv = sqrt((*stddev) / (imageDim * channel));

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

            Rpp32f *mean, *stddev;
            mean = batch_mean + batchCount;
            stddev = batch_stddev + batchCount;

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                #pragma omp critical
                                *mean = *mean + (Rpp32f) *srcPtrTemp;

                                srcPtrTemp++;
                            }
                        }
                    }
                }
            }

            *mean = *mean / (channel * imageDim);

            omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                #pragma omp critical
                                *stddev = *stddev + (((*mean) - (*srcPtrTemp)) * ((*mean) - (*srcPtrTemp)));

                                srcPtrTemp++;
                            }
                        }
                    }
                }
            }

            *stddsv = sqrt((*stddev) / (imageDim * channel));

        }
    }

    return RPP_SUCCESS;
}
*/

// template <typename T>
// RppStatus mean_stddev_host(T* srcPtr, RppiSize srcSize,
//                             Rpp32f *mean, Rpp32f *stddev,
//                             RppiChnFormat chnFormat, unsigned int channel)
// {
//     int i;
//     Rpp32f *meanTemp, *stdDevTemp;
//     meanTemp = mean;
//     stdDevTemp = stddev;
//     T* srcPtrTemp = srcPtr;
//     *meanTemp = 0;
//     *stdDevTemp = 0;
//     for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
//     {
//         *meanTemp += *srcPtr;
//         srcPtr++;
//     }
//     *meanTemp = (*meanTemp) / (srcSize.height * srcSize.width * channel);

//     for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
//     {
//         *stdDevTemp += (((*meanTemp) - (*srcPtrTemp)) * ((*meanTemp) - (*srcPtrTemp)));
//         srcPtrTemp++;
//     }
//     *stdDevTemp = sqrt((*stdDevTemp) / (srcSize.height * srcSize.width * channel));
//     return RPP_SUCCESS;
// }

/**************** integral ***************/

template <typename T, typename U>
RppStatus integral_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* batch_dstPtr,
                              Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtr = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        U *dstPtr = (U*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(U));

        compute_unpadded_from_padded_host(batch_srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtr,
                                          chnFormat, channel);
        compute_unpadded_from_padded_host(batch_dstPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtr,
                                          chnFormat, channel);

        RppiSize srcSize;
        srcSize.height = batch_srcSize[batchCount].height;
        srcSize.width = batch_srcSize[batchCount].width;

        U* srcPtrCopy = (U *)calloc(channel * srcSize.height * srcSize.width, sizeof(U));

        T *srcPtrTemp;
        srcPtrTemp = srcPtr;
        U *srcPtrCopyTemp;
        srcPtrCopyTemp = srcPtrCopy;

        for(int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            *srcPtrCopyTemp = (U) *srcPtrTemp;
            srcPtrTemp++;
            srcPtrCopyTemp++;
        }

        RppiSize srcSizeMod;
        srcSizeMod.height = srcSize.height + 1;
        srcSizeMod.width = srcSize.width + 1;

        U* srcPtrMod = (U *)calloc(channel * srcSizeMod.height * srcSizeMod.width, sizeof(U));

        generate_corner_padded_image_host(srcPtrCopy, srcSize, srcPtrMod, srcSizeMod, 1, chnFormat, channel);

        U *srcPtrModTemp, *srcPtrModTempAbove;
        U *dstPtrTemp;
        srcPtrModTemp = srcPtrMod;
        dstPtrTemp = dstPtr;

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrModTemp = srcPtrMod + (c * srcSizeMod.height * srcSizeMod.width) + srcSizeMod.width + 1;
                srcPtrModTempAbove = srcPtrModTemp - srcSizeMod.width;

                for (int i = 0; i < srcSize.height; i++)
                {
                    for (int j = 0; j < srcSize.width; j++)
                    {
                        *dstPtrTemp = *srcPtrModTemp + *(srcPtrModTemp - 1) + *srcPtrModTempAbove - *(srcPtrModTempAbove - 1);
                        *srcPtrModTemp = *dstPtrTemp;
                        dstPtrTemp++;
                        srcPtrModTemp++;
                        srcPtrModTempAbove++;
                    }
                    srcPtrModTemp++;
                    srcPtrModTempAbove++;
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            srcPtrModTemp = srcPtrMod + (channel * (srcSizeMod.width + 1));
            srcPtrModTempAbove = srcPtrModTemp - (channel * srcSizeMod.width);

            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrModTemp + *(srcPtrModTemp - channel) + *srcPtrModTempAbove - *(srcPtrModTempAbove - channel);
                        *srcPtrModTemp = *dstPtrTemp;
                        dstPtrTemp++;
                        srcPtrModTemp++;
                        srcPtrModTempAbove++;
                    }
                }
                srcPtrModTemp += channel;
                srcPtrModTempAbove += channel;
            }
        }

        compute_padded_from_unpadded_host(dstPtr, srcSize, batch_srcSizeMax[batchCount], batch_dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtr);
        free(dstPtr);
        free(srcPtrCopy);
        free(srcPtrMod);
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
RppStatus integral_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    U* srcPtrCopy = (U *)calloc(channel * srcSize.height * srcSize.width, sizeof(U));

    T *srcPtrTemp;
    srcPtrTemp = srcPtr;
    U *srcPtrCopyTemp;
    srcPtrCopyTemp = srcPtrCopy;

    for(int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *srcPtrCopyTemp = (U) *srcPtrTemp;
        srcPtrTemp++;
        srcPtrCopyTemp++;
    }

    RppiSize srcSizeMod;
    srcSizeMod.height = srcSize.height + 1;
    srcSizeMod.width = srcSize.width + 1;

    U* srcPtrMod = (U *)calloc(channel * srcSizeMod.height * srcSizeMod.width, sizeof(U));

    generate_corner_padded_image_host(srcPtrCopy, srcSize, srcPtrMod, srcSizeMod, 1, chnFormat, channel);

    U *srcPtrModTemp, *srcPtrModTempAbove;
    U *dstPtrTemp;
    srcPtrModTemp = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrModTemp = srcPtrMod + (c * srcSizeMod.height * srcSizeMod.width) + srcSizeMod.width + 1;
            srcPtrModTempAbove = srcPtrModTemp - srcSizeMod.width;

            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    *dstPtrTemp = *srcPtrModTemp + *(srcPtrModTemp - 1) + *srcPtrModTempAbove - *(srcPtrModTempAbove - 1);
                    *srcPtrModTemp = *dstPtrTemp;
                    dstPtrTemp++;
                    srcPtrModTemp++;
                    srcPtrModTempAbove++;
                }
                srcPtrModTemp++;
                srcPtrModTempAbove++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrModTemp = srcPtrMod + (channel * (srcSizeMod.width + 1));
        srcPtrModTempAbove = srcPtrModTemp - (channel * srcSizeMod.width);

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrModTemp + *(srcPtrModTemp - channel) + *srcPtrModTempAbove - *(srcPtrModTempAbove - channel);
                    *srcPtrModTemp = *dstPtrTemp;
                    dstPtrTemp++;
                    srcPtrModTemp++;
                    srcPtrModTempAbove++;
                }
            }
            srcPtrModTemp += channel;
            srcPtrModTempAbove += channel;
        }
    }

    free(srcPtrCopy);
    free(srcPtrMod);

    return RPP_SUCCESS;

}

// /**************** Histogram ***************/

// template <typename T>
// RppStatus histogram_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins,
//                          Rpp32u channel)
// {
//     histogram_kernel_host(srcPtr, srcSize, outputHistogram, bins - 1, channel);

//     return RPP_SUCCESS;

// }

// /**************** Mean & Standard Deviation ***************/
// It has been put in statistical shd move there

template <typename T>
RppStatus mean_stddev_host(T* srcPtr, RppiSize srcSize,
                            Rpp32f *mean, Rpp32f *stddev,
                            RppiChnFormat chnFormat, unsigned int channel)
{
    int i;
    Rpp32f *meanTemp, *stdDevTemp;
    meanTemp = mean;
    stdDevTemp = stddev;
    T* srcPtrTemp = srcPtr;
    *meanTemp = 0;
    *stdDevTemp = 0;
    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *meanTemp += *srcPtr;
        srcPtr++;
    }
    *meanTemp = (*meanTemp) / (srcSize.height * srcSize.width * channel);

    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *stdDevTemp += (((*meanTemp) - (*srcPtrTemp)) * ((*meanTemp) - (*srcPtrTemp)));
        srcPtrTemp++;
    }
    *stdDevTemp = sqrt((*stdDevTemp) / (srcSize.height * srcSize.width * channel));
    return RPP_SUCCESS;
}

#endif // #ifndef STATISTICAL_OPERATIONS_HPP
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

#ifndef HOST_ARITHMETIC_OPERATIONS_HPP
#define HOST_ARITHMETIC_OPERATIONS_HPP

#include "rpp_cpu_batchpd.hpp"

/**************** absolute_difference ***************/

template <typename T>
RppStatus absolute_difference_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
                    Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPABS(pixel);
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp =(T) pixel;

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
                Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPABS(pixel);
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp =(T) pixel;

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
RppStatus absolute_difference_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPABS(pixel);
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** accumulate_weighted ***************/

template <typename T>
RppStatus accumulate_weighted_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
                                         Rpp32f *batch_alpha,
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

            Rpp32f alpha = batch_alpha[batchCount];

            T *srcPtr1Image, *srcPtr2Image;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtr1Channel, *srcPtr2Channel;
                srcPtr1Channel = srcPtr1Image + (c * imageDimMax);
                srcPtr2Channel = srcPtr2Image + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtr1Temp, *srcPtr2Temp;
                    srcPtr1Temp = srcPtr1Channel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtr2Temp = srcPtr2Channel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        srcPtr1Temp += batch_srcSizeMax[batchCount].width;
                        srcPtr2Temp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                pixel = ((1 - alpha) * ((Rpp32f) (*srcPtr1Temp))) + (alpha * ((Rpp32f) (*srcPtr2Temp)));
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtr1Temp =(T) pixel;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                            }
                            else
                            {
                                srcPtr1Temp++;
                                srcPtr2Temp++;
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

            Rpp32f alpha = batch_alpha[batchCount];

            T *srcPtr1Image, *srcPtr2Image;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtr1Temp, *srcPtr2Temp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    srcPtr1Temp += elementsInRowMax;
                    srcPtr2Temp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            srcPtr1Temp += channel;
                            srcPtr2Temp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                pixel = ((1 - alpha) * ((Rpp32f) (*srcPtr1Temp))) + (alpha * ((Rpp32f) (*srcPtr2Temp)));
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtr1Temp =(T) pixel;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
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
RppStatus accumulate_weighted_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                   Rpp32f alpha,
                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        Rpp32f pixel = ((1 - alpha) * ((Rpp32f) (*srcPtr1Temp))) + (alpha * ((Rpp32f) (*srcPtr2Temp)));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }

    return RPP_SUCCESS;

}

/**************** accumulate ***************/

template <typename T>
RppStatus accumulate_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
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

            T *srcPtr1Image, *srcPtr2Image;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtr1Channel, *srcPtr2Channel;
                srcPtr1Channel = srcPtr1Image + (c * imageDimMax);
                srcPtr2Channel = srcPtr2Image + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32s pixel;

                    T *srcPtr1Temp, *srcPtr2Temp;
                    srcPtr1Temp = srcPtr1Channel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtr2Temp = srcPtr2Channel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        srcPtr1Temp += batch_srcSizeMax[batchCount].width;
                        srcPtr2Temp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtr1Temp = (T) pixel;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
                            }
                            else
                            {
                                srcPtr1Temp++;
                                srcPtr2Temp++;
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

            T *srcPtr1Image, *srcPtr2Image;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32s pixel;

                T *srcPtr1Temp, *srcPtr2Temp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    srcPtr1Temp += elementsInRowMax;
                    srcPtr2Temp += elementsInRowMax;
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            srcPtr1Temp += channel;
                            srcPtr2Temp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtr1Temp = (T) pixel;

                                srcPtr1Temp++;
                                srcPtr2Temp++;
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
RppStatus accumulate_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }

    return RPP_SUCCESS;

}

/**************** add ***************/

template <typename T>
RppStatus add_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
                    Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
                Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
RppStatus add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** subtract ***************/

template <typename T>
RppStatus subtract_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
                    Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
                Rpp32s pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
RppStatus subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_subtract_host(srcPtr1, srcPtr2, srcSize, dstPtr, channel);

    return RPP_SUCCESS;

}

/**************** magnitude ***************/

template <typename T>
RppStatus magnitude_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
                    Rpp32f pixel;

                    Rpp32s srcPtr1Value, srcPtr2Value;

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
                                srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                                srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                                pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) round(pixel);

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
                Rpp32f pixel;
                Rpp32s srcPtr1Value, srcPtr2Value;

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
                                srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                                srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                                pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) round(pixel);

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

template <typename T>
RppStatus magnitude_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_magnitude_host(srcPtr1, srcPtr2, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;

}

/**************** multiply ***************/

template <typename T>
RppStatus multiply_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
                    Rpp32f pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) * ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
                Rpp32f pixel;

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
                                pixel = ((Rpp32s) (*srcPtr1Temp)) * ((Rpp32s) (*srcPtr2Temp));
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp = (T) pixel;

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
RppStatus multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_multiply_host(srcPtr1, srcPtr2, srcSize, dstPtr, channel);

    return RPP_SUCCESS;

}

/**************** phase ***************/

template <typename T>
RppStatus phase_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                           RppiROI *roiPoints, Rpp32u nbatchSize,
                           RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32f multiplier = 255 / 1.570796;
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
                    Rpp32f pixel;

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
                                pixel = atan(((Rpp32f) (*srcPtr1Temp)) / ((Rpp32f) (*srcPtr2Temp)));
                                pixel = pixel * multiplier;
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp =(T) pixel;

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
                Rpp32f pixel;

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
                                pixel = atan(((Rpp32f) (*srcPtr1Temp)) / ((Rpp32f) (*srcPtr2Temp)));
                                pixel = pixel * multiplier;
                                pixel = RPPPIXELCHECK(pixel);
                                *dstPtrTemp =(T) pixel;

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
RppStatus phase_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
    Rpp32f multiplier = 255 / 1.570796;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = atan(((Rpp32f) (*srcPtr1Temp)) / ((Rpp32f) (*srcPtr2Temp)));
        pixel = pixel * multiplier;
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** accumulate_squared ***************/

template <typename T>
RppStatus accumulate_squared_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax,
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
                    Rpp32s pixel;

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
                            if((x1 <= j) && (j <= x2 ))
                            {
                                pixel = (Rpp32s) *srcPtrTemp;
                                pixel = pixel * pixel;
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtrTemp =(T) pixel;

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

            T *srcPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32s pixel;

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
                                pixel = (Rpp32s) *srcPtrTemp;
                                pixel = pixel * pixel;
                                pixel = RPPPIXELCHECK(pixel);
                                *srcPtrTemp =(T) pixel;

                                srcPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus accumulate_squared_host(T* srcPtr, RppiSize srcSize,
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = (Rpp32s) *srcPtrTemp;
        pixel = pixel * pixel;
        pixel = RPPPIXELCHECK(pixel);
        *srcPtrTemp =(T) pixel;
        srcPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Absolute Difference ***************/

// template <typename T, typename U>
// RppStatus absolute_difference_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
//                                    RppiChnFormat chnFormat, Rpp32u channel)
// {
//     T *srcPtr1Temp, *dstPtrTemp;
//     U *srcPtr2Temp;
//     srcPtr1Temp = srcPtr1;
//     srcPtr2Temp = srcPtr2;
//     dstPtrTemp = dstPtr;

//     Rpp32f pixel;

//     for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
//     {
//         pixel = ((Rpp32f) (*srcPtr1Temp)) - ((Rpp32f) (*srcPtr2Temp));
//         pixel = RPPABS(pixel);
//         pixel = RPPPIXELCHECK(pixel);
//         *dstPtrTemp =(T) pixel;
//         srcPtr1Temp++;
//         srcPtr2Temp++;
//         dstPtrTemp++;
//     }

//     return RPP_SUCCESS;

// }

// /**************** Accumulate Weighted ***************/

// template <typename T, typename U>
// RppStatus accumulate_weighted_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
//                                    Rpp32f alpha,
//                                    RppiChnFormat chnFormat, Rpp32u channel)
// {
//     T *srcPtr1Temp;
//     U *srcPtr2Temp;
//     srcPtr1Temp = srcPtr1;
//     srcPtr2Temp = srcPtr2;

//     Rpp32f pixel;

//     for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
//     {
//         Rpp32f pixel = ((1 - alpha) * ((Rpp32f) (*srcPtr1Temp))) + (alpha * ((Rpp32f) (*srcPtr2Temp)));
//         pixel = RPPPIXELCHECK(pixel);
//         *srcPtr1Temp =(T) pixel;
//         srcPtr1Temp++;
//         srcPtr2Temp++;
//     }

//     return RPP_SUCCESS;

// }

// /**************** Accumulate ***************/

// template <typename T, typename U>
// RppStatus accumulate_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
//                           RppiChnFormat chnFormat, Rpp32u channel)
// {
//     T *srcPtr1Temp;
//     U *srcPtr2Temp;
//     srcPtr1Temp = srcPtr1;
//     srcPtr2Temp = srcPtr2;

//     Rpp32f pixel;

//     for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
//     {
//         pixel = ((Rpp32f) (*srcPtr1Temp)) + ((Rpp32f) (*srcPtr2Temp));
//         pixel = RPPPIXELCHECK(pixel);
//         *srcPtr1Temp =(T) pixel;
//         srcPtr1Temp++;
//         srcPtr2Temp++;
//     }

//     return RPP_SUCCESS;

// }

// /**************** Add ***************/

// template <typename T, typename U>
// RppStatus add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
//                    RppiChnFormat chnFormat, Rpp32u channel)
// {
//     T *srcPtr1Temp, *dstPtrTemp;
//     U *srcPtr2Temp;
//     srcPtr1Temp = srcPtr1;
//     srcPtr2Temp = srcPtr2;
//     dstPtrTemp = dstPtr;

//     Rpp32f pixel;

//     for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
//     {
//         pixel = ((Rpp32f) (*srcPtr1Temp)) + ((Rpp32f) (*srcPtr2Temp));
//         pixel = RPPPIXELCHECK(pixel);
//         *dstPtrTemp =(T) pixel;
//         srcPtr1Temp++;
//         srcPtr2Temp++;
//         dstPtrTemp++;
//     }

//     return RPP_SUCCESS;

// }

// /**************** Subtract ***************/

// template <typename T, typename U>
// RppStatus subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
//                         RppiChnFormat chnFormat, Rpp32u channel)
// {
//     T *srcPtr1Temp, *dstPtrTemp;
//     U *srcPtr2Temp;
//     srcPtr1Temp = srcPtr1;
//     srcPtr2Temp = srcPtr2;
//     dstPtrTemp = dstPtr;

//     Rpp32f pixel;

//     for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
//     {
//         pixel = ((Rpp32f) (*srcPtr1Temp)) - ((Rpp32f) (*srcPtr2Temp));
//         pixel = RPPPIXELCHECK(pixel);
//         *dstPtrTemp =(T) pixel;
//         srcPtr1Temp++;
//         srcPtr2Temp++;
//         dstPtrTemp++;
//     }

//     return RPP_SUCCESS;

// }



/**************** Tensor Add ***************/

template <typename T, typename U>
RppStatus tensor_add_host(T* srcPtr1, U* srcPtr2, T* dstPtr,
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

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32s pixel;

    for (int i = 0; i < tensorSize; i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Tensor Subtract ***************/

template <typename T, typename U>
RppStatus tensor_subtract_host(T* srcPtr1, U* srcPtr2, T* dstPtr,
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

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32s pixel;

    for (int i = 0; i < tensorSize; i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Tensor Multiply ***************/

template <typename T, typename U>
RppStatus tensor_multiply_host(T* srcPtr1, U* srcPtr2, T* dstPtr,
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

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32s pixel;

    for (int i = 0; i < tensorSize; i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) * ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

#endif // #ifndef HOST_ARITHMETIC_OPERATIONS_HPP
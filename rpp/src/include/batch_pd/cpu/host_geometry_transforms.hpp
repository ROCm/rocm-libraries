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

#ifndef HOST_GEOMETRIC_TRASFORMS_HPP
#define HOST_GEOMETRIC_TRASFORMS_HPP

#include "rpp_cpu_batchpd.hpp"
#include "rpp_cpu_simd_math.hpp"

/**************** flip ***************/

template <typename T>
RppStatus flip_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                          Rpp32u *batch_flipAxis, RppiROI *roiPoints,
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32u flipAxis = batch_flipAxis[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            if (flipAxis == RPPI_HORIZONTAL_AXIS)
            {
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y2 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            memcpy(dstPtrTemp, srcPtrChannelROI, roiPoints[batchCount].roiWidth * sizeof(T));
                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            dstPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI -= batch_srcSizeMax[batchCount].width;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
                        }
                    }
                }
            }
            else if (flipAxis == RPPI_VERTICAL_AXIS)
            {
                Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + roiPoints[batchCount].roiWidth;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            __m128i px0;
                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                srcPtrChannelROI -= 15;
                                px0 = _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                                px0 = _mm_shuffle_epi8(px0, vMask);
                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                srcPtrChannelROI -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                            }

                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI += srcROIIncrement;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
                        }
                    }
                }
            }
            else if (flipAxis == RPPI_BOTH_AXIS)
            {
                Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - roiPoints[batchCount].roiWidth;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y2 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            __m128i px0;
                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                srcPtrChannelROI -= 15;
                                px0 = _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                                px0 = _mm_shuffle_epi8(px0, vMask);
                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                srcPtrChannelROI -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                            }

                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI -= srcROIIncrement;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32u flipAxis = batch_flipAxis[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * roiPoints[batchCount].roiWidth;

            if (flipAxis == RPPI_HORIZONTAL_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y2 * elementsInRowMax) + (x1 * channel);


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
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        memcpy(dstPtrTemp, srcPtrROI, elementsInRowROI * sizeof(T));
                        srcPtrTemp += elementsInRowROI;
                        dstPtrTemp += elementsInRowROI;
                        srcPtrROI -= elementsInRowMax;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
            else if (flipAxis == RPPI_VERTICAL_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * elementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u bufferLength = elementsInRowROI;
                Rpp32u alignedLength = (bufferLength / 15) * 15;

                Rpp32u srcROIIncrement = elementsInRowMax + elementsInRowROI;


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
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            srcPtrROI -= 13;
                            px0 = _mm_loadu_si128((__m128i *)srcPtrROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrROI -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                            dstPtrTemp += channel;
                            srcPtrROI -= channel;
                        }

                        srcPtrTemp += elementsInRowROI;
                        srcPtrROI += srcROIIncrement;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
            else if (flipAxis == RPPI_BOTH_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y2 * elementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u bufferLength = elementsInRowROI;
                Rpp32u alignedLength = (bufferLength / 15) * 15;

                Rpp32u srcROIIncrement = elementsInRowMax - elementsInRowROI;


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
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            srcPtrROI -= 13;
                            px0 = _mm_loadu_si128((__m128i *)srcPtrROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrROI -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                            dstPtrTemp += channel;
                            srcPtrROI -= channel;
                        }

                        srcPtrTemp += elementsInRowROI;
                        srcPtrROI -= srcROIIncrement;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus flip_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u flipAxis,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c + 1) * srcSize.height * srcSize.width) - srcSize.width;
                for (int i = 0; i < srcSize.height; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    dstPtrTemp += srcSize.width;
                    srcPtrTemp -= srcSize.width;
                }
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            T *srcPtrTemp2;
            srcPtrTemp2 = srcPtr;

            Rpp32u bufferLength = srcSize.width;
            Rpp32u alignedLength = (bufferLength / 16) * 16;

            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width) + srcSize.width - 1;
                for (int i = 0; i < srcSize.height; i++)
                {
                    srcPtrTemp2 = srcPtrTemp;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrTemp2 -= 15;
                        px0 = _mm_loadu_si128((__m128i *)srcPtrTemp2);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrTemp2 -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) *srcPtrTemp2--;
                    }
                    srcPtrTemp = srcPtrTemp + srcSize.width;
                }
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            T *srcPtrTemp2;
            srcPtrTemp2 = srcPtr;

            Rpp32u bufferLength = srcSize.width;
            Rpp32u alignedLength = (bufferLength / 16) * 16;

            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c+1) * srcSize.height * srcSize.width) - 1;
                for (int i = 0; i < srcSize.height; i++)
                {
                    srcPtrTemp2 = srcPtrTemp;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrTemp2 -= 15;
                        px0 = _mm_loadu_si128((__m128i *)srcPtrTemp2);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrTemp2 -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) *srcPtrTemp2--;
                    }
                    srcPtrTemp = srcPtrTemp - srcSize.width;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * ((srcSize.height * srcSize.width) - srcSize.width));
            for (int i = 0; i < srcSize.height; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                dstPtrTemp += elementsInRow;
                srcPtrTemp -= elementsInRow;
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            Rpp32u bufferLength = channel * srcSize.width;
            Rpp32u alignedLength = (bufferLength / 15) * 15;

            srcPtrTemp = srcPtr + (channel * (srcSize.width - 1));
            for (int i = 0; i < srcSize.height; i++)
            {
                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrTemp -= 13;
                    px0 = _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px0 = _mm_shuffle_epi8(px0, vMask);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrTemp -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrTemp -= channel;
                }

                srcPtrTemp = srcPtrTemp + (channel * (2 * srcSize.width));
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            Rpp32u bufferLength = channel * srcSize.width;
            Rpp32u alignedLength = (bufferLength / 15) * 15;

            srcPtrTemp = srcPtr + (channel * ((srcSize.height * srcSize.width) - 1));
            for (int i = 0; i < srcSize.height; i++)
            {
                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrTemp -= 13;
                    px0 = _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px0 = _mm_shuffle_epi8(px0, vMask);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrTemp -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrTemp -= channel;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** fisheye ***************/

template <typename T>
RppStatus fisheye_base_host(T* srcPtrTemp, RppiSize srcSize, T* dstPtrTemp,
                            Rpp64u j, Rpp32u elementsPerChannel, Rpp32u elements,
                            Rpp32f newI, Rpp32f newIsquared,
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f newJ, newIsrc, newJsrc, newJsquared, euclideanDistance, newEuclideanDistance, theta;
    int iSrc, jSrc, srcPosition;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
        newJsquared = newJ * newJ;
        euclideanDistance = sqrt(newIsquared + newJsquared);
        if (euclideanDistance >= 0 && euclideanDistance <= 1)
        {
            newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
            newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;

            if (newEuclideanDistance <= 1.0)
            {
                theta = atan2(newI, newJ);

                newIsrc = newEuclideanDistance * sin(theta);
                newJsrc = newEuclideanDistance * cos(theta);

                iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                srcPosition = (int)((iSrc * srcSize.width) + jSrc);

                if ((srcPosition >= 0) && (srcPosition < elementsPerChannel))
                {
                    *dstPtrTemp++ = *(srcPtrTemp + srcPosition);
                }
                else
                {
                    *dstPtrTemp++ = (T) 0;
                }
            }
            else
            {
                *dstPtrTemp++ = (T) 0;
            }
        }
        else
        {
            *dstPtrTemp++ = (T) 0;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
        newJsquared = newJ * newJ;
        euclideanDistance = sqrt(newIsquared + newJsquared);
        if (euclideanDistance >= 0 && euclideanDistance <= 1)
        {
            newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
            newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;

            if (newEuclideanDistance <= 1.0)
            {
                theta = atan2(newI, newJ);

                newIsrc = newEuclideanDistance * sin(theta);
                newJsrc = newEuclideanDistance * cos(theta);

                iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                srcPosition = (int)(channel * ((iSrc * srcSize.width) + jSrc));

                if ((srcPosition >= 0) && (srcPosition < elements))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp++ = *(srcPtrTemp + srcPosition + c);
                    }
                }
                else
                {
                    memset(dstPtrTemp, 0, 3 * sizeof(T));
                    dstPtrTemp += 3;
                }
            }
            else
            {
                memset(dstPtrTemp, 0, 3 * sizeof(T));
                dstPtrTemp += 3;
            }
        }
        else
        {
            memset(dstPtrTemp, 0, 3 * sizeof(T));
            dstPtrTemp += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus fisheye_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiROI *roiPoints,
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32f newI, newIsquared;
            Rpp32u elementsPerChannelMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsMax = channel * elementsPerChannelMax;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32s srcPositionArrayInt[4] = {0};
            Rpp32f eD[4] = {0};
            Rpp32f nED[4] = {0};

            __m128i px0, px1;
            __m128 p0, p1, p2;
            __m128 q0, pCmp1, pCmp2, pMask;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul2 = _mm_set1_ps(0.5);
            __m128 pMul3 = _mm_set1_ps(halfHeight);
            __m128 pMul4 = _mm_set1_ps(halfWidth);
            __m128 pWidthMax = _mm_set1_ps((Rpp32f) batch_srcSizeMax[batchCount].width);
            __m128i pxSrcPosition = _mm_set1_epi32(0);

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtrTemp2 = srcPtrChannel;

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                        Rpp32f newIsquared = newI * newI;

                        __m128 pNewI = _mm_set1_ps(newI);
                        __m128 pNewIsquared = _mm_set1_ps(newIsquared);

                        Rpp64u vectorLoopCount = x1;
                        for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                        {
                            pMask = _mm_set1_ps(1.0);

                            p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                            p0 = _mm_mul_ps(p0, pMul);
                            p0 = _mm_sub_ps(p0, pOne);

                            p1 = _mm_mul_ps(p0, p0);
                            p1 = _mm_add_ps(pNewIsquared, p1);
                            p1 = _mm_sqrt_ps(p1);

                            pCmp1 = _mm_cmpge_ps(p1, pZero);
                            pCmp2 = _mm_cmple_ps(p1, pOne);
                            pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                            _mm_storeu_ps(eD, pMask);

                            if(eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                            {
                                p2 = _mm_mul_ps(p1, p1);
                                p2 = _mm_sub_ps(pOne, p2);
                                p2 = _mm_sqrt_ps(p2);
                                p2 = _mm_sub_ps(pOne, p2);
                                p2 = _mm_add_ps(p1, p2);
                                p2 = _mm_mul_ps(p2, pMul2);

                                _mm_storeu_ps(nED, p2);

                                if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                                {
                                    q0 = atan2_ps(pNewI, p0);

                                    sincos_ps(q0, &p0, &p1);

                                    p0 = _mm_mul_ps(p2, p0);
                                    p1 = _mm_mul_ps(p2, p1);

                                    p0 = _mm_add_ps(p0, pOne);
                                    p0 = _mm_mul_ps(p0, pMul3);
                                    p1 = _mm_add_ps(p1, pOne);
                                    p1 = _mm_mul_ps(p1, pMul4);

                                    p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidthMax);

                                    px0 = _mm_cvtps_epi32(p0);
                                    px1 = _mm_cvtps_epi32(_mm_floor_ps(p1));

                                    pxSrcPosition = _mm_add_epi32(px0, px1);

                                    _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                                    for (int pos = 0; pos < 4; pos++)
                                    {
                                        if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elementsPerChannelMax))
                                        {
                                            *dstPtrTemp = *(srcPtrTemp2 + srcPositionArrayInt[pos]);
                                        }
                                        else
                                        {
                                            *dstPtrTemp = (T) 0;
                                        }
                                        dstPtrTemp++;
                                    }
                                }
                                else
                                {
                                    for (int id = 0; id < 4; id++)
                                    {
                                        if (nED[id] <= 1.0)
                                        {
                                            fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                            dstPtrTemp++;
                                        }
                                        else
                                        {
                                            *dstPtrTemp++ = (T) 0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int id = 0; id < 4; id++)
                                {
                                    if (eD[id] != 0.0)
                                    {
                                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                        dstPtrTemp++;
                                    }
                                    else
                                    {
                                        *dstPtrTemp++ = (T) 0;
                                    }
                                }
                            }
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                            dstPtrTemp++;
                        }

                        srcPtrTemp += bufferLength;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32f newI, newIsquared;
            Rpp32u elementsPerChannelMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsMax = channel * elementsPerChannelMax;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u elementsInBuffer = channel * bufferLength;

            Rpp32s srcPositionArrayInt[4] = {0};
            Rpp32f eD[4] = {0};
            Rpp32f nED[4] = {0};

            __m128i px0, px1;
            __m128 p0, p1, p2;
            __m128 q0, pCmp1, pCmp2, pMask;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pThree = _mm_set1_ps(3.0);
            __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul2 = _mm_set1_ps(0.5);
            __m128 pMul3 = _mm_set1_ps(halfHeight);
            __m128 pMul4 = _mm_set1_ps(halfWidth);
            __m128 pWidthMax = _mm_set1_ps((Rpp32f) elementsInRowMax);
            __m128i pxSrcPosition = _mm_set1_epi32(0);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                Rpp32f newIsquared = newI * newI;

                T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                srcPtrTemp2 = srcPtrImage;

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                    Rpp32f newIsquared = newI * newI;

                    __m128 pNewI = _mm_set1_ps(newI);
                    __m128 pNewIsquared = _mm_set1_ps(newIsquared);

                    Rpp64u vectorLoopCount = x1;
                    for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                    {
                        pMask = _mm_set1_ps(1.0);

                        p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                        p0 = _mm_mul_ps(p0, pMul);
                        p0 = _mm_sub_ps(p0, pOne);

                        p1 = _mm_mul_ps(p0, p0);
                        p1 = _mm_add_ps(pNewIsquared, p1);
                        p1 = _mm_sqrt_ps(p1);

                        pCmp1 = _mm_cmpge_ps(p1, pZero);
                        pCmp2 = _mm_cmple_ps(p1, pOne);
                        pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                        _mm_storeu_ps(eD, pMask);

                        if (eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                        {
                            p2 = _mm_mul_ps(p1, p1);
                            p2 = _mm_sub_ps(pOne, p2);
                            p2 = _mm_sqrt_ps(p2);
                            p2 = _mm_sub_ps(pOne, p2);
                            p2 = _mm_add_ps(p1, p2);
                            p2 = _mm_mul_ps(p2, pMul2);

                            _mm_storeu_ps(nED, p2);

                            if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                            {
                                q0 = atan2_ps(pNewI, p0);

                                sincos_ps(q0, &p0, &p1);

                                p0 = _mm_mul_ps(p2, p0);
                                p1 = _mm_mul_ps(p2, p1);

                                p0 = _mm_add_ps(p0, pOne);
                                p0 = _mm_mul_ps(p0, pMul3);
                                p1 = _mm_add_ps(p1, pOne);
                                p1 = _mm_mul_ps(p1, pMul4);

                                p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidthMax);

                                px0 = _mm_cvtps_epi32(p0);
                                px1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_floor_ps(p1), pThree));

                                pxSrcPosition = _mm_add_epi32(px0, px1);

                                _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                                for (int pos = 0; pos < 4; pos++)
                                {
                                    if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elementsMax))
                                    {
                                        for (int c = 0; c < channel; c++)
                                        {
                                            *dstPtrTemp++ = *(srcPtrTemp2 + srcPositionArrayInt[pos] + c);
                                        }
                                    }
                                    else
                                    {
                                        memset(dstPtrTemp, 0, 3 * sizeof(T));
                                        dstPtrTemp += 3;
                                    }
                                }
                            }
                            else
                            {
                                for (int id = 0; id < 4; id++)
                                {
                                    if (nED[id] <= 1.0)
                                    {
                                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                        dstPtrTemp += 3;
                                    }
                                    else
                                    {
                                        memset(dstPtrTemp, 0, 3 * sizeof(T));
                                        dstPtrTemp += 3;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int id = 0; id < 4; id++)
                            {
                                if (eD[id] != 0.0)
                                {
                                    fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                    dstPtrTemp += 3;
                                }
                                else
                                {
                                    memset(dstPtrTemp, 0, 3 * sizeof(T));
                                    dstPtrTemp += 3;
                                }
                            }
                        }
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                        dstPtrTemp += 3;
                    }

                    srcPtrTemp += elementsInBuffer;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus fisheye_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f newI, newIsquared;
    Rpp32u elementsPerChannel = srcSize.height * srcSize.width;
    Rpp32u elements = channel * elementsPerChannel;

    Rpp32f halfHeight = srcSize.height / 2;
    Rpp32f halfWidth = srcSize.width / 2;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u bufferLength = srcSize.width;
        Rpp32u alignedLength = (bufferLength / 4) * 4;

        Rpp32s srcPositionArrayInt[4] = {0};
        Rpp32f eD[4] = {0};
        Rpp32f nED[4] = {0};

        __m128i px0, px1;
        __m128 p0, p1, p2;
        __m128 q0, pCmp1, pCmp2, pMask;
        __m128 pZero = _mm_set1_ps(0.0);
        __m128 pOne = _mm_set1_ps(1.0);
        __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) srcSize.width);
        __m128 pMul2 = _mm_set1_ps(0.5);
        __m128 pMul3 = _mm_set1_ps(halfHeight);
        __m128 pMul4 = _mm_set1_ps(halfWidth);
        __m128 pWidth = _mm_set1_ps((Rpp32f) srcSize.width);
        __m128i pxSrcPosition = _mm_set1_epi32(0);

        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for(int i = 0; i < srcSize.height; i++)
            {
                newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
                newIsquared = newI * newI;

                __m128 pNewI = _mm_set1_ps(newI);
                __m128 pNewIsquared = _mm_set1_ps(newIsquared);

                Rpp64u vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pMask = _mm_set1_ps(1.0);

                    p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                    p0 = _mm_mul_ps(p0, pMul);
                    p0 = _mm_sub_ps(p0, pOne);

                    p1 = _mm_mul_ps(p0, p0);
                    p1 = _mm_add_ps(pNewIsquared, p1);
                    p1 = _mm_sqrt_ps(p1);

                    pCmp1 = _mm_cmpge_ps(p1, pZero);
                    pCmp2 = _mm_cmple_ps(p1, pOne);
                    pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                    _mm_storeu_ps(eD, pMask);

                    if(eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                    {
                        p2 = _mm_mul_ps(p1, p1);
                        p2 = _mm_sub_ps(pOne, p2);
                        p2 = _mm_sqrt_ps(p2);
                        p2 = _mm_sub_ps(pOne, p2);
                        p2 = _mm_add_ps(p1, p2);
                        p2 = _mm_mul_ps(p2, pMul2);

                        _mm_storeu_ps(nED, p2);

                        if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                        {
                            q0 = atan2_ps(pNewI, p0);

                            sincos_ps(q0, &p0, &p1);

                            p0 = _mm_mul_ps(p2, p0);
                            p1 = _mm_mul_ps(p2, p1);

                            p0 = _mm_add_ps(p0, pOne);
                            p0 = _mm_mul_ps(p0, pMul3);
                            p1 = _mm_add_ps(p1, pOne);
                            p1 = _mm_mul_ps(p1, pMul4);

                            p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidth);

                            px0 = _mm_cvtps_epi32(p0);
                            px1 = _mm_cvtps_epi32(_mm_floor_ps(p1));

                            pxSrcPosition = _mm_add_epi32(px0, px1);

                            _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                            for (int pos = 0; pos < 4; pos++)
                            {
                                if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elementsPerChannel))
                                {
                                    *dstPtrTemp = *(srcPtrTemp + srcPositionArrayInt[pos]);
                                }
                                else
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                dstPtrTemp++;
                            }
                        }
                        else
                        {
                            for (int id = 0; id < 4; id++)
                            {
                                if (nED[id] <= 1.0)
                                {
                                    fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                                    dstPtrTemp++;
                                }
                                else
                                {
                                    *dstPtrTemp++ = (T) 0;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int id = 0; id < 4; id++)
                        {
                            if (eD[id] != 0.0)
                            {
                                fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp++ = (T) 0;
                            }
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32f elementsInRow = channel * srcSize.width;

        Rpp32u bufferLength = srcSize.width;
        Rpp32u alignedLength = (bufferLength / 4) * 4;

        Rpp32s srcPositionArrayInt[4] = {0};
        Rpp32f eD[4] = {0};
        Rpp32f nED[4] = {0};

        __m128i px0, px1;
        __m128 p0, p1, p2;
        __m128 q0, pCmp1, pCmp2, pMask;
        __m128 pZero = _mm_set1_ps(0.0);
        __m128 pOne = _mm_set1_ps(1.0);
        __m128 pThree = _mm_set1_ps(3.0);
        __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) srcSize.width);
        __m128 pMul2 = _mm_set1_ps(0.5);
        __m128 pMul3 = _mm_set1_ps(halfHeight);
        __m128 pMul4 = _mm_set1_ps(halfWidth);
        __m128 pWidth = _mm_set1_ps(elementsInRow);
        __m128i pxSrcPosition = _mm_set1_epi32(0);

        for (int i = 0; i < srcSize.height; i++)
        {
            newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
            newIsquared = newI * newI;

            __m128 pNewI = _mm_set1_ps(newI);
            __m128 pNewIsquared = _mm_set1_ps(newIsquared);

            Rpp64u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                pMask = _mm_set1_ps(1.0);

                p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                p0 = _mm_mul_ps(p0, pMul);
                p0 = _mm_sub_ps(p0, pOne);

                p1 = _mm_mul_ps(p0, p0);
                p1 = _mm_add_ps(pNewIsquared, p1);
                p1 = _mm_sqrt_ps(p1);

                pCmp1 = _mm_cmpge_ps(p1, pZero);
                pCmp2 = _mm_cmple_ps(p1, pOne);
                pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                _mm_storeu_ps(eD, pMask);

                if (eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                {
                    p2 = _mm_mul_ps(p1, p1);
                    p2 = _mm_sub_ps(pOne, p2);
                    p2 = _mm_sqrt_ps(p2);
                    p2 = _mm_sub_ps(pOne, p2);
                    p2 = _mm_add_ps(p1, p2);
                    p2 = _mm_mul_ps(p2, pMul2);

                    _mm_storeu_ps(nED, p2);

                    if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                    {
                        q0 = atan2_ps(pNewI, p0);

                        sincos_ps(q0, &p0, &p1);

                        p0 = _mm_mul_ps(p2, p0);
                        p1 = _mm_mul_ps(p2, p1);

                        p0 = _mm_add_ps(p0, pOne);
                        p0 = _mm_mul_ps(p0, pMul3);
                        p1 = _mm_add_ps(p1, pOne);
                        p1 = _mm_mul_ps(p1, pMul4);

                        p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidth);

                        px0 = _mm_cvtps_epi32(p0);
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_floor_ps(p1), pThree));

                        pxSrcPosition = _mm_add_epi32(px0, px1);

                        _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                        for (int pos = 0; pos < 4; pos++)
                        {
                            if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elements))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp++ = *(srcPtrTemp + srcPositionArrayInt[pos] + c);
                                }
                            }
                            else
                            {
                                memset(dstPtrTemp, 0, 3 * sizeof(T));
                                dstPtrTemp += 3;
                            }
                        }
                    }
                    else
                    {
                        for (int id = 0; id < 4; id++)
                        {
                            if (nED[id] <= 1.0)
                            {
                                fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                                dstPtrTemp += 3;
                            }
                            else
                            {
                                memset(dstPtrTemp, 0, 3 * sizeof(T));
                                dstPtrTemp += 3;
                            }
                        }
                    }
                }
                else
                {
                    for (int id = 0; id < 4; id++)
                    {
                        if (eD[id] != 0.0)
                        {
                            fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                            dstPtrTemp += 3;
                        }
                        else
                        {
                            memset(dstPtrTemp, 0, 3 * sizeof(T));
                            dstPtrTemp += 3;
                        }
                    }
                }
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                fisheye_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount, elementsPerChannel, elements, newI, newIsquared, chnFormat, channel);
                dstPtrTemp += 3;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** lens_correction ***************/

template <typename T>
RppStatus lens_correction_base_host(T* srcPtrTemp, RppiSize srcSize, T* dstPtrTemp,
                            Rpp64u j, Rpp32u heightLimit, Rpp32u widthLimit, Rpp32f halfHeight, Rpp32f halfWidth,
                            Rpp32f newIsquared, Rpp32f newIZoom, Rpp32f invCorrectionRadius, Rpp32f zoom, Rpp32u elementsInRow,
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f newJ, euclideanDistance, correctedDistance, theta;
    Rpp32f srcLocationRow, srcLocationColumn;
    Rpp32u srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTopRow, *srcPtrBottomRow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        newJ = j - halfWidth;
        euclideanDistance = sqrt(newIsquared + newJ * newJ);
        correctedDistance = euclideanDistance * invCorrectionRadius;
        theta = atan(correctedDistance) / correctedDistance;

        srcLocationRow = halfHeight + theta * newIZoom;
        srcLocationColumn = halfWidth + theta * newJ * zoom;

        if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) &&
            (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
        {
            srcLocationRowFloor = (Rpp32u) RPPFLOOR(srcLocationRow);
            srcLocationColumnFloor = (Rpp32u) RPPFLOOR(srcLocationColumn);

            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
            if (srcLocationRowFloor > heightLimit)
            {
                srcLocationRowFloor = heightLimit;
            }

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
            srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

            Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
            if (srcLocationColumnFloor > widthLimit)
            {
                srcLocationColumnFloor = widthLimit;
            }

            *dstPtrTemp = (T) (((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                    + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                    + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                    + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth)));
        }
        else
        {
            *dstPtrTemp = (T) 0;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        newJ = j - halfWidth;
        euclideanDistance = sqrt(newIsquared + newJ * newJ);
        correctedDistance = euclideanDistance * invCorrectionRadius;
        theta = atan(correctedDistance) / correctedDistance;

        srcLocationRow = halfHeight + theta * newIZoom;
        srcLocationColumn = halfWidth + theta * newJ * zoom;

        if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) &&
            (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
        {
            srcLocationRowFloor = (Rpp32u) RPPFLOOR(srcLocationRow);
            srcLocationColumnFloor = (Rpp32u) RPPFLOOR(srcLocationColumn);

            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
            if (srcLocationRowFloor > heightLimit)
            {
                srcLocationRowFloor = heightLimit;
            }

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
            if (srcLocationColumnFloor > widthLimit)
            {
                srcLocationColumnFloor = widthLimit;
            }

            srcLocationColumnFloor = srcLocationColumnFloor * channel;

            for (int c = 0; c < channel; c++)
            {
                *dstPtrTemp++ = (T) (((*(srcPtrTopRow + c + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                  + ((*(srcPtrTopRow + c + srcLocationColumnFloor + channel)) * (1 - weightedHeight) * (weightedWidth))
                                  + ((*(srcPtrBottomRow + c + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                  + ((*(srcPtrBottomRow + c + srcLocationColumnFloor + channel)) * (weightedHeight) * (weightedWidth)));
            }
        }
        else
        {
            memset(dstPtrTemp, 0, 3 * sizeof(T));
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus lens_correction_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                     Rpp32f *batch_strength, Rpp32f *batch_zoom, RppiROI *roiPoints,
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32f strength = batch_strength[batchCount];
            Rpp32f zoom = batch_zoom[batchCount];

            Rpp32f newI, newIsquared, newIZoom;

            T *srcPtrTopRow, *srcPtrBottomRow;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;
            Rpp32u heightLimit = batch_srcSize[batchCount].height - 2;
            Rpp32u widthLimit = batch_srcSize[batchCount].width - 2;

            if (strength == 0) strength = 0.000001;

            Rpp32f invCorrectionRadius = 1.0 / (sqrt(batch_srcSize[batchCount].height * batch_srcSize[batchCount].height + batch_srcSize[batchCount].width * batch_srcSize[batchCount].width) / strength);

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32f mask[4] = {0};
            Rpp32u srcLocRF[4] = {0};
            Rpp32u srcLocCF[4] = {0};
            Rpp32f param1[4] = {0};
            Rpp32f param2[4] = {0};
            Rpp32f param3[4] = {0};
            Rpp32f param4[4] = {0};

            __m128i px0, px1, px3, px4;
            __m128 p0, p1, p2, p3, p4, p5, p6, p7, pMask;
            __m128 q0, pCmp1, pCmp2;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pAdd1 = _mm_set1_ps(halfHeight);
            __m128 pAdd2 = _mm_set1_ps(halfWidth);
            __m128 pHeight = _mm_set1_ps((Rpp32f) batch_srcSize[batchCount].height);
            __m128 pWidth = _mm_set1_ps((Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul = _mm_set1_ps(invCorrectionRadius);

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtrTemp2 = srcPtrChannel;

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        newI = i - halfHeight;
                        newIsquared = newI * newI;
                        newIZoom = newI * zoom;

                        __m128 pNewI = _mm_set1_ps(newI);
                        __m128 pNewIsquared = _mm_set1_ps(newIsquared);
                        __m128 pZoom = _mm_set1_ps(zoom);
                        __m128 pNewIZoom = _mm_set1_ps(newIZoom);

                        Rpp64u vectorLoopCount = x1;
                        for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                        {
                            pMask = _mm_set1_ps(1.0);

                            p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                            p0 = _mm_sub_ps(p0, pAdd2);

                            __m128 pNewJZoom = _mm_mul_ps(p0, pZoom);

                            p1 = _mm_add_ps(pNewIsquared, _mm_mul_ps(p0, p0));
                            p1 = _mm_sqrt_ps(p1);
                            p1 = _mm_mul_ps(p1, pMul);

                            q0 = _mm_div_ps(atan_ps(p1), p1);

                            p1 = _mm_mul_ps(q0, pNewIZoom);
                            p1 = _mm_add_ps(pAdd1, p1);
                            p2 = _mm_mul_ps(q0, pNewJZoom);
                            p2 = _mm_add_ps(pAdd2, p2);

                            pCmp1 = _mm_and_ps(_mm_cmpge_ps(p1, pZero), _mm_cmplt_ps(p1, pHeight));
                            pCmp2 = _mm_and_ps(_mm_cmpge_ps(p2, pZero), _mm_cmplt_ps(p2, pWidth));
                            pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                            _mm_storeu_ps(mask, pMask);

                            if(mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0)
                            {
                                p3 = _mm_floor_ps(p1);
                                p4 = _mm_floor_ps(p2);

                                p1 = _mm_sub_ps(p1, p3);
                                p2 = _mm_sub_ps(p2, p4);
                                p0 = _mm_mul_ps(p1, p2);

                                p5 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(pOne, p2), p1), p0);
                                p6 = _mm_sub_ps(p2, p0);
                                p7 = _mm_sub_ps(p1, p0);

                                px3 = _mm_cvtps_epi32(p3);
                                px4 = _mm_cvtps_epi32(p4);

                                _mm_storeu_si128((__m128i *)srcLocRF, px3);
                                _mm_storeu_si128((__m128i *)srcLocCF, px4);

                                _mm_storeu_ps(param1, p5);
                                _mm_storeu_ps(param2, p6);
                                _mm_storeu_ps(param3, p7);
                                _mm_storeu_ps(param4, p0);

                                for (int pos = 0; pos < 4; pos++)
                                {
                                    if (srcLocRF[pos] > heightLimit)
                                    {
                                        srcLocRF[pos] = heightLimit;
                                    }
                                    if (srcLocCF[pos] > widthLimit)
                                    {
                                        srcLocCF[pos] = widthLimit;
                                    }

                                    srcPtrTopRow = srcPtrTemp2 + srcLocRF[pos] * elementsInRowMax;
                                    srcPtrBottomRow  = srcPtrTopRow + elementsInRowMax;

                                    *dstPtrTemp++ = (T) (((*(srcPtrTopRow + srcLocCF[pos])) * param1[pos])
                                                    + ((*(srcPtrTopRow + srcLocCF[pos] + 1)) * param2[pos])
                                                    + ((*(srcPtrBottomRow + srcLocCF[pos])) * param3[pos])
                                                    + ((*(srcPtrBottomRow + srcLocCF[pos] + 1)) * param4[pos]));
                                }
                            }
                            else
                            {
                                for (int id = 0; id < 4; id++)
                                {
                                    if (mask[id] != 0)
                                    {
                                        lens_correction_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRowMax, chnFormat, channel);
                                        dstPtrTemp++;
                                    }
                                    else
                                    {
                                        *dstPtrTemp++ = (T) 0;
                                    }
                                }
                            }
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            lens_correction_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRowMax, chnFormat, channel);
                            dstPtrTemp++;
                        }

                        srcPtrTemp += bufferLength;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32f strength = batch_strength[batchCount];
            Rpp32f zoom = batch_zoom[batchCount];

            Rpp32f newI, newIsquared, newIZoom;

            T *srcPtrTopRow, *srcPtrBottomRow;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;
            Rpp32u heightLimit = batch_srcSize[batchCount].height - 2;
            Rpp32u widthLimit = batch_srcSize[batchCount].width - 2;

            if (strength == 0) strength = 0.000001;

            Rpp32f invCorrectionRadius = 1.0 / (sqrt(batch_srcSize[batchCount].height * batch_srcSize[batchCount].height + batch_srcSize[batchCount].width * batch_srcSize[batchCount].width) / strength);

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u elementsInBuffer = channel * bufferLength;

            Rpp32f mask[4] = {0};
            Rpp32u srcLocRF[4] = {0};
            Rpp32u srcLocCF[4] = {0};
            Rpp32f param1[4] = {0};
            Rpp32f param2[4] = {0};
            Rpp32f param3[4] = {0};
            Rpp32f param4[4] = {0};

            __m128i px0, px1, px3, px4;
            __m128 p0, p1, p2, p3, p4, p5, p6, p7, pMask;
            __m128 q0, pCmp1, pCmp2;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pAdd1 = _mm_set1_ps(halfHeight);
            __m128 pAdd2 = _mm_set1_ps(halfWidth);
            __m128 pHeight = _mm_set1_ps((Rpp32f) batch_srcSize[batchCount].height);
            __m128 pWidth = _mm_set1_ps((Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul = _mm_set1_ps(invCorrectionRadius);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                srcPtrTemp2 = srcPtrImage;

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    newI = i - halfHeight;
                    newIsquared = newI * newI;
                    newIZoom = newI * zoom;

                    __m128 pNewI = _mm_set1_ps(newI);
                    __m128 pNewIsquared = _mm_set1_ps(newIsquared);
                    __m128 pZoom = _mm_set1_ps(zoom);
                    __m128 pNewIZoom = _mm_set1_ps(newIZoom);

                    Rpp64u vectorLoopCount = x1;
                    for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                    {
                        pMask = _mm_set1_ps(1.0);

                        p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                        p0 = _mm_sub_ps(p0, pAdd2);

                        __m128 pNewJZoom = _mm_mul_ps(p0, pZoom);

                        p1 = _mm_add_ps(pNewIsquared, _mm_mul_ps(p0, p0));
                        p1 = _mm_sqrt_ps(p1);
                        p1 = _mm_mul_ps(p1, pMul);

                        q0 = _mm_div_ps(atan_ps(p1), p1);

                        p1 = _mm_mul_ps(q0, pNewIZoom);
                        p1 = _mm_add_ps(pAdd1, p1);
                        p2 = _mm_mul_ps(q0, pNewJZoom);
                        p2 = _mm_add_ps(pAdd2, p2);

                        pCmp1 = _mm_and_ps(_mm_cmpge_ps(p1, pZero), _mm_cmplt_ps(p1, pHeight));
                        pCmp2 = _mm_and_ps(_mm_cmpge_ps(p2, pZero), _mm_cmplt_ps(p2, pWidth));
                        pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                        _mm_storeu_ps(mask, pMask);

                        if(mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0)
                        {
                            p3 = _mm_floor_ps(p1);
                            p4 = _mm_floor_ps(p2);

                            p1 = _mm_sub_ps(p1, p3);
                            p2 = _mm_sub_ps(p2, p4);
                            p0 = _mm_mul_ps(p1, p2);

                            p5 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(pOne, p2), p1), p0);
                            p6 = _mm_sub_ps(p2, p0);
                            p7 = _mm_sub_ps(p1, p0);

                            px3 = _mm_cvtps_epi32(p3);
                            px4 = _mm_cvtps_epi32(p4);

                            _mm_storeu_si128((__m128i *)srcLocRF, px3);
                            _mm_storeu_si128((__m128i *)srcLocCF, px4);

                            _mm_storeu_ps(param1, p5);
                            _mm_storeu_ps(param2, p6);
                            _mm_storeu_ps(param3, p7);
                            _mm_storeu_ps(param4, p0);

                            for (int pos = 0; pos < 4; pos++)
                            {
                                if (srcLocRF[pos] > heightLimit)
                                {
                                    srcLocRF[pos] = heightLimit;
                                }
                                if (srcLocCF[pos] > widthLimit)
                                {
                                    srcLocCF[pos] = widthLimit;
                                }

                                srcLocCF[pos] *= channel;

                                srcPtrTopRow = srcPtrTemp2 + srcLocRF[pos] * elementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + elementsInRowMax;

                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp++ = (T) (((*(srcPtrTopRow + c + srcLocCF[pos])) * param1[pos])
                                                        + ((*(srcPtrTopRow + c + srcLocCF[pos] + channel)) * param2[pos])
                                                        + ((*(srcPtrBottomRow + c + srcLocCF[pos])) * param3[pos])
                                                        + ((*(srcPtrBottomRow + c + srcLocCF[pos] + channel)) * param4[pos]));
                                }
                            }
                        }
                        else
                        {
                            for (int id = 0; id < 4; id++)
                            {
                                if (mask[id] != 0)
                                {
                                    lens_correction_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRowMax, chnFormat, channel);
                                    dstPtrTemp += 3;
                                }
                                else
                                {
                                    memset(dstPtrTemp, 0, 3 * sizeof(T));
                                    dstPtrTemp += 3;
                                }
                            }
                        }
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        lens_correction_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRowMax, chnFormat, channel);
                        dstPtrTemp += 3;
                    }

                    srcPtrTemp += elementsInBuffer;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus lens_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                               Rpp32f strength, Rpp32f zoom,
                               RppiChnFormat chnFormat, Rpp32u channel)
{
    if (strength < 0 || zoom < 1)
    {
        return RPP_ERROR;
    }

    Rpp32u heightLimit, widthLimit;
    Rpp32f halfHeight, halfWidth, newI, newIsquared, invCorrectionRadius, newIZoom;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32u elementsInRow = srcSize.width * channel;

    halfHeight = ((Rpp32f) srcSize.height) / 2.0;
    halfWidth = ((Rpp32f) srcSize.width) / 2.0;
    heightLimit = srcSize.height - 2;
    widthLimit = srcSize.width - 2;

    if (strength == 0) strength = 0.000001;

    invCorrectionRadius = 1.0 / (sqrt(srcSize.height * srcSize.height + srcSize.width * srcSize.width) / strength);

    Rpp32u bufferLength = srcSize.width;
    Rpp32u alignedLength = (bufferLength / 4) * 4;

    Rpp32f mask[4] = {0};
    Rpp32u srcLocRF[4] = {0};
    Rpp32u srcLocCF[4] = {0};
    Rpp32f param1[4] = {0};
    Rpp32f param2[4] = {0};
    Rpp32f param3[4] = {0};
    Rpp32f param4[4] = {0};

    __m128i px0, px1, px3, px4;
    __m128 p0, p1, p2, p3, p4, p5, p6, p7, pMask;
    __m128 q0, pCmp1, pCmp2;
    __m128 pZero = _mm_set1_ps(0.0);
    __m128 pOne = _mm_set1_ps(1.0);
    __m128 pAdd1 = _mm_set1_ps(halfHeight);
    __m128 pAdd2 = _mm_set1_ps(halfWidth);
    __m128 pHeight = _mm_set1_ps((Rpp32f) srcSize.height);
    __m128 pWidth = _mm_set1_ps((Rpp32f) srcSize.width);
    __m128 pMul = _mm_set1_ps(invCorrectionRadius);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                newI = i - halfHeight;
                newIsquared = newI * newI;
                newIZoom = newI * zoom;

                __m128 pNewI = _mm_set1_ps(newI);
                __m128 pNewIsquared = _mm_set1_ps(newIsquared);
                __m128 pZoom = _mm_set1_ps(zoom);
                __m128 pNewIZoom = _mm_set1_ps(newIZoom);

                Rpp64u vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pMask = _mm_set1_ps(1.0);

                    p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                    p0 = _mm_sub_ps(p0, pAdd2);

                    __m128 pNewJZoom = _mm_mul_ps(p0, pZoom);

                    p1 = _mm_add_ps(pNewIsquared, _mm_mul_ps(p0, p0));
                    p1 = _mm_sqrt_ps(p1);
                    p1 = _mm_mul_ps(p1, pMul);

                    q0 = _mm_div_ps(atan_ps(p1), p1);

                    p1 = _mm_mul_ps(q0, pNewIZoom);
                    p1 = _mm_add_ps(pAdd1, p1);
                    p2 = _mm_mul_ps(q0, pNewJZoom);
                    p2 = _mm_add_ps(pAdd2, p2);

                    pCmp1 = _mm_and_ps(_mm_cmpge_ps(p1, pZero), _mm_cmplt_ps(p1, pHeight));
                    pCmp2 = _mm_and_ps(_mm_cmpge_ps(p2, pZero), _mm_cmplt_ps(p2, pWidth));
                    pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                    _mm_storeu_ps(mask, pMask);

                    if(mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0)
                    {
                        p3 = _mm_floor_ps(p1);
                        p4 = _mm_floor_ps(p2);

                        p1 = _mm_sub_ps(p1, p3);
                        p2 = _mm_sub_ps(p2, p4);
                        p0 = _mm_mul_ps(p1, p2);

                        p5 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(pOne, p2), p1), p0);
                        p6 = _mm_sub_ps(p2, p0);
                        p7 = _mm_sub_ps(p1, p0);

                        px3 = _mm_cvtps_epi32(p3);
                        px4 = _mm_cvtps_epi32(p4);

                        _mm_storeu_si128((__m128i *)srcLocRF, px3);
                        _mm_storeu_si128((__m128i *)srcLocCF, px4);

                        _mm_storeu_ps(param1, p5);
                        _mm_storeu_ps(param2, p6);
                        _mm_storeu_ps(param3, p7);
                        _mm_storeu_ps(param4, p0);

                        for (int pos = 0; pos < 4; pos++)
                        {
                            if (srcLocRF[pos] > heightLimit)
                            {
                                srcLocRF[pos] = heightLimit;
                            }
                            if (srcLocCF[pos] > widthLimit)
                            {
                                srcLocCF[pos] = widthLimit;
                            }

                            srcPtrTopRow = srcPtrTemp + srcLocRF[pos] * srcSize.width;
                            srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                            *dstPtrTemp++ = (T) (((*(srcPtrTopRow + srcLocCF[pos])) * param1[pos])
                                              + ((*(srcPtrTopRow + srcLocCF[pos] + 1)) * param2[pos])
                                              + ((*(srcPtrBottomRow + srcLocCF[pos])) * param3[pos])
                                              + ((*(srcPtrBottomRow + srcLocCF[pos] + 1)) * param4[pos]));
                        }
                    }
                    else
                    {
                        for (int id = 0; id < 4; id++)
                        {
                            if (mask[id] != 0)
                            {
                                lens_correction_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRow, chnFormat, channel);
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp++ = (T) 0;
                            }
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    lens_correction_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRow, chnFormat, channel);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            newI = i - halfHeight;
            newIsquared = newI * newI;
            newIZoom = newI * zoom;

            __m128 pNewI = _mm_set1_ps(newI);
            __m128 pNewIsquared = _mm_set1_ps(newIsquared);
            __m128 pZoom = _mm_set1_ps(zoom);
            __m128 pNewIZoom = _mm_set1_ps(newIZoom);

            Rpp64u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                pMask = _mm_set1_ps(1.0);

                p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                p0 = _mm_sub_ps(p0, pAdd2);

                __m128 pNewJZoom = _mm_mul_ps(p0, pZoom);

                p1 = _mm_add_ps(pNewIsquared, _mm_mul_ps(p0, p0));
                p1 = _mm_sqrt_ps(p1);
                p1 = _mm_mul_ps(p1, pMul);

                q0 = _mm_div_ps(atan_ps(p1), p1);

                p1 = _mm_mul_ps(q0, pNewIZoom);
                p1 = _mm_add_ps(pAdd1, p1);
                p2 = _mm_mul_ps(q0, pNewJZoom);
                p2 = _mm_add_ps(pAdd2, p2);

                pCmp1 = _mm_and_ps(_mm_cmpge_ps(p1, pZero), _mm_cmplt_ps(p1, pHeight));
                pCmp2 = _mm_and_ps(_mm_cmpge_ps(p2, pZero), _mm_cmplt_ps(p2, pWidth));
                pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                _mm_storeu_ps(mask, pMask);

                if(mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0)
                {
                    p3 = _mm_floor_ps(p1);
                    p4 = _mm_floor_ps(p2);

                    p1 = _mm_sub_ps(p1, p3);
                    p2 = _mm_sub_ps(p2, p4);
                    p0 = _mm_mul_ps(p1, p2);

                    p5 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(pOne, p2), p1), p0);
                    p6 = _mm_sub_ps(p2, p0);
                    p7 = _mm_sub_ps(p1, p0);

                    px3 = _mm_cvtps_epi32(p3);
                    px4 = _mm_cvtps_epi32(p4);

                    _mm_storeu_si128((__m128i *)srcLocRF, px3);
                    _mm_storeu_si128((__m128i *)srcLocCF, px4);

                    _mm_storeu_ps(param1, p5);
                    _mm_storeu_ps(param2, p6);
                    _mm_storeu_ps(param3, p7);
                    _mm_storeu_ps(param4, p0);

                    for (int pos = 0; pos < 4; pos++)
                    {
                        if (srcLocRF[pos] > heightLimit)
                        {
                            srcLocRF[pos] = heightLimit;
                        }
                        if (srcLocCF[pos] > widthLimit)
                        {
                            srcLocCF[pos] = widthLimit;
                        }

                        srcLocCF[pos] *= channel;

                        srcPtrTopRow = srcPtrTemp + srcLocRF[pos] * elementsInRow;
                        srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp++ = (T) (((*(srcPtrTopRow + c + srcLocCF[pos])) * param1[pos])
                                                + ((*(srcPtrTopRow + c + srcLocCF[pos] + channel)) * param2[pos])
                                                + ((*(srcPtrBottomRow + c + srcLocCF[pos])) * param3[pos])
                                                + ((*(srcPtrBottomRow + c + srcLocCF[pos] + channel)) * param4[pos]));
                        }
                    }
                }
                else
                {
                    for (int id = 0; id < 4; id++)
                    {
                        if (mask[id] != 0)
                        {
                            lens_correction_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount + id, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRow, chnFormat, channel);
                            dstPtrTemp += 3;
                        }
                        else
                        {
                            memset(dstPtrTemp, 0, 3 * sizeof(T));
                            dstPtrTemp += 3;
                        }
                    }
                }
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                lens_correction_base_host(srcPtrTemp, srcSize, dstPtrTemp, vectorLoopCount, heightLimit, widthLimit, halfHeight, halfWidth, newIsquared, newIZoom, invCorrectionRadius, zoom, elementsInRow, chnFormat, channel);
                dstPtrTemp += 3;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** scale ***************/

template <typename T>
RppStatus scale_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                             Rpp32f *batch_percentage, RppiROI *roiPoints,
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

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f percentage = batch_percentage[batchCount] / 100;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f srcLocationRow = ((Rpp32f) i) / percentage;
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                        T *srcPtrTopRow, *srcPtrBottomRow;
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                        srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f srcLocationColumn = ((Rpp32f) j) / percentage;
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else
                                {
                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
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

            Rpp32f percentage = batch_percentage[batchCount] / 100;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;


            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32f srcLocationRow = ((Rpp32f) i) / percentage;
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                    T *srcPtrTopRow, *srcPtrBottomRow;
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                    srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f srcLocationColumn = ((Rpp32f) j) / percentage;
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                            Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else
                            {
                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
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
RppStatus scale_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f percentage,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }
    if (percentage < 0)
    {
        return RPP_ERROR;
    }

    percentage /= 100;

    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRow = ((Rpp32f) i) / percentage;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / percentage;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = (T) 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / percentage;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            for (int j = 0; j < dstSize.width; j++)
            {
                srcLocationColumn = ((Rpp32f) j) / percentage;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** rotate ***************/

template <typename T>
RppStatus rotate_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                             Rpp32f *batch_angleDeg, RppiROI *roiPoints,
                             Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
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

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f angleDeg = batch_angleDeg[batchCount];
            Rpp32f angleRad = -RAD(angleDeg);
            Rpp32f rotate[4] = {0};
            rotate[0] = cos(angleRad);
            rotate[1] = sin(angleRad);
            rotate[2] = -sin(angleRad);
            rotate[3] = cos(angleRad);
            Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);

            Rpp32f halfSrcHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfSrcWidth = batch_srcSize[batchCount].width / 2;
            Rpp32f halfDstHeight = batch_dstSize[batchCount].height / 2;
            Rpp32f halfDstWidth = batch_dstSize[batchCount].width / 2;
            Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
            Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

            Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
            Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
            Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
            Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f srcLocationRowTerm1 = -rotate[3] * i;
                        Rpp32f srcLocationColumnTerm1 = rotate[2] * i;

                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                                Rpp32f srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;

                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
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

            Rpp32f angleDeg = batch_angleDeg[batchCount];
            Rpp32f angleRad = -RAD(angleDeg);
            Rpp32f rotate[4] = {0};
            rotate[0] = cos(angleRad);
            rotate[1] = sin(angleRad);
            rotate[2] = -sin(angleRad);
            rotate[3] = cos(angleRad);
            Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);

            Rpp32f halfSrcHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfSrcWidth = batch_srcSize[batchCount].width / 2;
            Rpp32f halfDstHeight = batch_dstSize[batchCount].height / 2;
            Rpp32f halfDstWidth = batch_dstSize[batchCount].width / 2;
            Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
            Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

            Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
            Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
            Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
            Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;


            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32f srcLocationRowTerm1 = -rotate[3] * i;
                    Rpp32f srcLocationColumnTerm1 = rotate[2] * i;

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                            Rpp32f srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus rotate_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f angleDeg,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f angleRad = -RAD(angleDeg);
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);

    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);
    Rpp32f srcLocationRow, srcLocationColumn, srcLocationRowTerm1, srcLocationColumnTerm1, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;

    Rpp32f halfSrcHeight = srcSize.height / 2;
    Rpp32f halfSrcWidth = srcSize.width / 2;
    Rpp32f halfDstHeight = dstSize.height / 2;
    Rpp32f halfDstWidth = dstSize.width / 2;
    Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
    Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

    Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
    Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
    Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
    Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRowTerm1 = -rotate[3] * i;
                srcLocationColumnTerm1 = rotate[2] * i;
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                    srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRowTerm1 = -rotate[3] * i;
            srcLocationColumnTerm1 = rotate[2] * i;
            for (int j = 0; j < dstSize.width; j++)
            {
                srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                    dstPtrTemp += channel;
                }
                else
                {
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** resize ***************/

template <typename T, typename U>
RppStatus resize_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                            RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            U *dstPtrROI = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            if (outputFormatToggle == 1)
            {
                U *dstPtrROICopy = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROICopy, dstSize, chnFormat, channel);
                compute_planar_to_packed_host(dstPtrROICopy, dstSize, dstPtrROI, channel);
                if ((typeid(Rpp8u) == typeid(T)) && (typeid(Rpp8u) != typeid(U)))
                {
                    normalize_kernel_host(dstPtrROI, dstSize, channel);
                }
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROICopy);
            }
            else
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
                if ((typeid(Rpp8u) == typeid(T)) && (typeid(Rpp8u) != typeid(U)))
                {
                    normalize_kernel_host(dstPtrROI, dstSize, channel);
                }
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            U *dstPtrROI = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            if (outputFormatToggle == 1)
            {
                U *dstPtrROICopy = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROICopy, dstSize, chnFormat, channel);
                compute_packed_to_planar_host(dstPtrROICopy, dstSize, dstPtrROI, channel);
                if ((typeid(Rpp8u) == typeid(T)) && (typeid(Rpp8u) != typeid(U)))
                {
                    normalize_kernel_host(dstPtrROI, dstSize, channel);
                }
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROICopy);
            }
            else
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
                if ((typeid(Rpp8u) == typeid(T)) && (typeid(Rpp8u) != typeid(U)))
                {
                    normalize_kernel_host(dstPtrROI, dstSize, channel);
                }
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus resize_u8_i8_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                            RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROIipType = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            U *dstPtrROI = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROIipType, dstSize, chnFormat, channel);

            T *dstPtrROIipTypeTemp;
            dstPtrROIipTypeTemp = dstPtrROIipType;

            U *dstPtrROITemp;
            dstPtrROITemp = dstPtrROI;

            for (int i = 0; i < (dstSize.height * dstSize.width * channel); i++)
            {
                *dstPtrROITemp = (U) (((Rpp32s) *dstPtrROIipTypeTemp) - 128);
                dstPtrROITemp++;
                dstPtrROIipTypeTemp++;
            }

            if (outputFormatToggle == 1)
            {
                U *dstPtrROI2 = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
                compute_planar_to_packed_host(dstPtrROI, dstSize, dstPtrROI2, channel);
                compute_padded_from_unpadded_host(dstPtrROI2, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROI2);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
            free(dstPtrROIipType);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
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

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROIipType = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            U *dstPtrROI = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROIipType, dstSize, chnFormat, channel);

            T *dstPtrROIipTypeTemp;
            dstPtrROIipTypeTemp = dstPtrROIipType;

            U *dstPtrROITemp;
            dstPtrROITemp = dstPtrROI;

            for (int i = 0; i < (dstSize.height * dstSize.width * channel); i++)
            {
                *dstPtrROITemp = (U) (((Rpp32s) *dstPtrROIipTypeTemp) - 128);
                dstPtrROITemp++;
                dstPtrROIipTypeTemp++;
            }

            if (outputFormatToggle == 1)
            {
                U *dstPtrROI2 = (U *)calloc(dstSize.height * dstSize.width * channel, sizeof(U));
                compute_packed_to_planar_host(dstPtrROI, dstSize, dstPtrROI2, channel);
                compute_padded_from_unpadded_host(dstPtrROI2, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROI2);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
            free(dstPtrROIipType);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    resize_kernel_host(srcPtr, srcSize, dstPtr, dstSize, chnFormat, channel);

    return RPP_SUCCESS;
}

// /**************** resize_crop ***************/

template <typename T>
RppStatus resize_crop_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                 Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2,
                                 Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                 RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrROICopy = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
                compute_planar_to_packed_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrROICopy = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
                compute_packed_to_planar_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_crop_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    resize_crop_kernel_host(srcPtr, srcSize, dstPtr, dstSize, x1, y1, x2, y2, chnFormat, channel);

    return RPP_SUCCESS;

}

/**************** warp_affine ***************/

template <typename T>
RppStatus warp_affine_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, RppiROI *roiPoints,
                                 Rpp32f *batch_affine,
                                 Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                 RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
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

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f affine[6] = {0};
            for (int i = 0; i < 6; i++)
            {
                affine[i] = batch_affine[(batchCount * 6) + i];
            }

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);

                                Rpp32f srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                                Rpp32f srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                                srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                                srcLocationRow += (batch_srcSize[batchCount].height / 2);

                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
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

            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f affine[6] = {0};
            for (int i = 0; i < 6; i++)
            {
                affine[i] = batch_affine[(batchCount * 6) + i];
            }

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;


            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);

                            Rpp32f srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                            Rpp32f srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                            srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                            srcLocationRow += (batch_srcSize[batchCount].height / 2);

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus warp_affine_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f* affine,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                Rpp32s iNew = i - (srcSize.height / 2);
                for (int j = 0; j < dstSize.width; j++)
                {
                    Rpp32s jNew = j - (srcSize.width / 2);

                    srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                    srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                    srcLocationColumn += (srcSize.width / 2);
                    srcLocationRow += (srcSize.height / 2);

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }

                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            Rpp32s iNew = i - (srcSize.height / 2);
            for (int j = 0; j < dstSize.width; j++)
            {
                Rpp32s jNew = j - (srcSize.width / 2);

                srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                srcLocationColumn += (srcSize.width / 2);
                srcLocationRow += (srcSize.height / 2);

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                    dstPtrTemp += channel;
                }
                else
                {
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** warp_perspective ***************/

template <typename T>
RppStatus warp_perspective_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                      RppiROI *roiPoints, Rpp32f *batch_perspective,
                                      Rpp32u nbatchSize,
                                      RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    //Rpp32f perspective[9] = {0.707, 0.707, 0, -0.707, 0.707, 0, 0.001, 0.001, 1};
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
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

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f perspective[9] = {0};
            for (int i = 0; i < 9; i++)
            {
                perspective[i] = batch_perspective[(batchCount * 9) + i];
            }

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);

                                Rpp32f srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                                Rpp32f srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                                srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                                srcLocationRow += (batch_srcSize[batchCount].height / 2);

                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
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

            Rpp32f perspective[9] = {0};
            for (int i = 0; i < 9; i++)
            {
                perspective[i] = batch_perspective[(batchCount * 9) + i];
            }

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSizeMax[batchCount].width;


            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);

                            Rpp32f srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                            Rpp32f srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                            srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                            srcLocationRow += (batch_srcSize[batchCount].height / 2);

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
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
RppStatus warp_perspective_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f* perspective,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                Rpp32s iNew = i - (srcSize.height / 2);
                for (int j = 0; j < dstSize.width; j++)
                {
                    Rpp32s jNew = j - (srcSize.width / 2);

                    srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                    srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                    srcLocationColumn += (srcSize.width / 2);
                    srcLocationRow += (srcSize.height / 2);

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }

                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            Rpp32s iNew = i - (srcSize.height / 2);
            for (int j = 0; j < dstSize.width; j++)
            {
                Rpp32s jNew = j - (srcSize.width / 2);

                srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                srcLocationColumn += (srcSize.width / 2);
                srcLocationRow += (srcSize.height / 2);

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

#endif
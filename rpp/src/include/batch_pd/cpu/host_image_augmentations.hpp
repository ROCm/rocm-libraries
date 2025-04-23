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

#ifndef HOST_IMAGE_AUGMENTATIONS_HPP
#define HOST_IMAGE_AUGMENTATIONS_HPP

#include <stdlib.h>
#include <time.h>

#include "rpp_cpu_batchpd.hpp"

/************ brightness ************/

template <typename T>
RppStatus brightness_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                Rpp32f *batch_alpha, Rpp32f *batch_beta,
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

            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

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
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(alpha);
                        __m128 pAdd = _mm_set1_ps(beta);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1, px2, px3;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_mul_ps(p0, pMul);
                            p1 = _mm_mul_ps(p1, pMul);
                            p2 = _mm_mul_ps(p2, pMul);
                            p3 = _mm_mul_ps(p3, pMul);
                            px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
                            px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
                            px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
                            px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                        }

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

            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

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
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(alpha);
                    __m128 pAdd = _mm_set1_ps(beta);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_mul_ps(p0, pMul);
                        p1 = _mm_mul_ps(p1, pMul);
                        p2 = _mm_mul_ps(p2, pMul);
                        p3 = _mm_mul_ps(p3, pMul);
                        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
                        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
                        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
                        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                    }

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
RppStatus brightness_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                          Rpp32f alpha, Rpp32f beta,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(alpha), pAdd = _mm_set1_ps(beta);
    __m128 p0, p1, p2, p3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

        p0 = _mm_mul_ps(p0, pMul);
        p1 = _mm_mul_ps(p1, pMul);
        p2 = _mm_mul_ps(p2, pMul);
        p3 = _mm_mul_ps(p3, pMul);
        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

        srcPtrTemp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
    }

    return RPP_SUCCESS;
}

// /**************** contrast ***************/

template <typename T>
RppStatus contrast_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32u *batch_new_min, Rpp32u *batch_new_max,
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

            Rpp32f new_min = batch_new_min[batchCount];
            Rpp32f new_max = batch_new_max[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T min[3] = {255, 255, 255};
            T max[3] = {0, 0, 0};

            if (channel == 1)
            {
                compute_1_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], &min[0], &max[0], chnFormat, channel);
            }
            else if (channel == 3)
            {
                compute_3_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], min, max, chnFormat, channel);
            }

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32f contrastFactor = (Rpp32f) (new_max - new_min) / (max[c] - min[c]);


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
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pContrastFactor = _mm_set1_ps(contrastFactor);
                        __m128i pMin = _mm_set1_epi16(min[c]);
                        __m128i pNewMin = _mm_set1_epi16(new_min);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                            px1 = _mm_max_epi16(_mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin), zero);    // pixels 8-15
                            px0 = _mm_max_epi16(_mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin), zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_mul_ps(p0, pContrastFactor);
                            p1 = _mm_mul_ps(p1, pContrastFactor);
                            p2 = _mm_mul_ps(p2, pContrastFactor);
                            p3 = _mm_mul_ps(p3, pContrastFactor);

                            px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p1));
                            px1 = _mm_packus_epi32(_mm_cvtps_epi32(p2), _mm_cvtps_epi32(p3));

                            px0 = _mm_add_epi16(px0, pNewMin);
                            px1 = _mm_add_epi16(px1, pNewMin);

                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));

                            srcPtrTemp += 16;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[c]) * contrastFactor) + new_min);
                        }

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

            Rpp32f new_min = batch_new_min[batchCount];
            Rpp32f new_max = batch_new_max[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T min[3] = {255, 255, 255};
            T max[3] = {0, 0, 0};

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            compute_3_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], min, max, chnFormat, channel);

            Rpp32f contrastFactorR = (new_max - new_min) / ((Rpp32f) (max[0] - min[0]));
            Rpp32f contrastFactorG = (new_max - new_min) / ((Rpp32f) (max[1] - min[1]));
            Rpp32f contrastFactorB = (new_max - new_min) / ((Rpp32f) (max[2] - min[2]));


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

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

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pzero = _mm_set1_ps(0.0);
                    __m128 pContrastFactor0 = _mm_setr_ps(contrastFactorR, contrastFactorG, contrastFactorB, contrastFactorR);
                    __m128 pContrastFactor1 = _mm_setr_ps(contrastFactorG, contrastFactorB, contrastFactorR, contrastFactorG);
                    __m128 pContrastFactor2 = _mm_setr_ps(contrastFactorB, contrastFactorR, contrastFactorG, contrastFactorB);

                    __m128 pMin0 = _mm_setr_ps((Rpp32f) min[0], (Rpp32f) min[1], (Rpp32f) min[2], (Rpp32f) min[0]);
                    __m128 pMin1 = _mm_setr_ps((Rpp32f) min[1], (Rpp32f) min[2], (Rpp32f) min[0], (Rpp32f) min[1]);
                    __m128 pMin2 = _mm_setr_ps((Rpp32f) min[2], (Rpp32f) min[0], (Rpp32f) min[1], (Rpp32f) min[2]);
                    __m128 pNewMin = _mm_set1_ps(new_min);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_max_ps(_mm_sub_ps(p0, pMin0), pzero);
                        p1 = _mm_max_ps(_mm_sub_ps(p1, pMin1), pzero);
                        p2 = _mm_max_ps(_mm_sub_ps(p2, pMin2), pzero);
                        p3 = _mm_max_ps(_mm_sub_ps(p3, pMin0), pzero);

                        p0 = _mm_mul_ps(p0, pContrastFactor0);
                        p1 = _mm_mul_ps(p1, pContrastFactor1);
                        p2 = _mm_mul_ps(p2, pContrastFactor2);
                        p3 = _mm_mul_ps(p3, pContrastFactor0);

                        p0 = _mm_add_ps(p0, pNewMin);
                        p1 = _mm_add_ps(p1, pNewMin);
                        p2 = _mm_add_ps(p2, pNewMin);
                        p3 = _mm_add_ps(p3, pNewMin);

                        px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p1));
                        px1 = _mm_packus_epi32(_mm_cvtps_epi32(p2), _mm_cvtps_epi32(p3));

                        _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));

                        srcPtrTemp += 15;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[0]) * contrastFactorR) + new_min);
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[1]) * contrastFactorG) + new_min);
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[2]) * contrastFactorB) + new_min);
                    }

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
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        Rpp32u new_min, Rpp32u new_max,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    T min[3] = {255, 255, 255};
    T max[3] = {0, 0, 0};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (channel == 1)
        {
            compute_1_channel_minmax_host(srcPtr, srcSize, srcSize, &min[0], &max[0], chnFormat, channel);
        }
        else if (channel == 3)
        {
            compute_3_channel_minmax_host(srcPtr, srcSize, srcSize, min, max, chnFormat, channel);
        }

        for (int c = 0; c < channel; c++)
        {
            Rpp32f contrastFactor = (Rpp32f) (new_max - new_min) / (max[c] - min[c]);

            int bufferLength = srcSize.height * srcSize.width;
            int alignedLength = (bufferLength / 16) * 16;

            __m128i const zero = _mm_setzero_si128();
            __m128 pContrastFactor = _mm_set1_ps(contrastFactor);
            __m128i pMin = _mm_set1_epi16(min[c]);
            __m128i pNewMin = _mm_set1_epi16(new_min);
            __m128 p0, p1, p2, p3;
            __m128i px0, px1;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                px1 = _mm_max_epi16(_mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin), zero);    // pixels 8-15
                px0 = _mm_max_epi16(_mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin), zero);    // pixels 0-7
                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                p0 = _mm_mul_ps(p0, pContrastFactor);
                p1 = _mm_mul_ps(p1, pContrastFactor);
                p2 = _mm_mul_ps(p2, pContrastFactor);
                p3 = _mm_mul_ps(p3, pContrastFactor);

                px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p1));
                px1 = _mm_packus_epi32(_mm_cvtps_epi32(p2), _mm_cvtps_epi32(p3));

                px0 = _mm_add_epi16(px0, pNewMin);
                px1 = _mm_add_epi16(px1, pNewMin);

                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));

                srcPtrTemp += 16;
                dstPtrTemp += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[c]) * contrastFactor) + new_min);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        compute_3_channel_minmax_host(srcPtr, srcSize, srcSize, min, max, chnFormat, channel);

        Rpp32f contrastFactorR = (Rpp32f) (new_max - new_min) / (max[0] - min[0]);
        Rpp32f contrastFactorG = (Rpp32f) (new_max - new_min) / (max[1] - min[1]);
        Rpp32f contrastFactorB = (Rpp32f) (new_max - new_min) / (max[2] - min[2]);

        int bufferLength = channel * srcSize.height * srcSize.width;
        int alignedLength = ((bufferLength / 15) * 15) - 1;

        __m128i const zero = _mm_setzero_si128();
        __m128 pzero = _mm_set1_ps(0.0);
        __m128 pContrastFactor0 = _mm_setr_ps(contrastFactorR, contrastFactorG, contrastFactorB, contrastFactorR);
        __m128 pContrastFactor1 = _mm_setr_ps(contrastFactorG, contrastFactorB, contrastFactorR, contrastFactorG);
        __m128 pContrastFactor2 = _mm_setr_ps(contrastFactorB, contrastFactorR, contrastFactorG, contrastFactorB);

        __m128 pMin0 = _mm_setr_ps((Rpp32f) min[0], (Rpp32f) min[1], (Rpp32f) min[2], (Rpp32f) min[0]);
        __m128 pMin1 = _mm_setr_ps((Rpp32f) min[1], (Rpp32f) min[2], (Rpp32f) min[0], (Rpp32f) min[1]);
        __m128 pMin2 = _mm_setr_ps((Rpp32f) min[2], (Rpp32f) min[0], (Rpp32f) min[1], (Rpp32f) min[2]);
        __m128 pNewMin = _mm_set1_ps(new_min);
        __m128 p0, p1, p2, p3;
        __m128i px0, px1;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

            p0 = _mm_max_ps(_mm_sub_ps(p0, pMin0), pzero);
            p1 = _mm_max_ps(_mm_sub_ps(p1, pMin1), pzero);
            p2 = _mm_max_ps(_mm_sub_ps(p2, pMin2), pzero);
            p3 = _mm_max_ps(_mm_sub_ps(p3, pMin0), pzero);

            p0 = _mm_mul_ps(p0, pContrastFactor0);
            p1 = _mm_mul_ps(p1, pContrastFactor1);
            p2 = _mm_mul_ps(p2, pContrastFactor2);
            p3 = _mm_mul_ps(p3, pContrastFactor0);

            p0 = _mm_add_ps(p0, pNewMin);
            p1 = _mm_add_ps(p1, pNewMin);
            p2 = _mm_add_ps(p2, pNewMin);
            p3 = _mm_add_ps(p3, pNewMin);

            px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p1));
            px1 = _mm_packus_epi32(_mm_cvtps_epi32(p2), _mm_cvtps_epi32(p3));

            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));

            srcPtrTemp += 15;
            dstPtrTemp += 15;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
        {
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[0]) * contrastFactorR) + new_min);
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[1]) * contrastFactorG) + new_min);
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[2]) * contrastFactorB) + new_min);
        }
    }

    return RPP_SUCCESS;
}

/************ blend ************/

template <typename T>
RppStatus blend_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            Rpp32f alpha = batch_alpha[batchCount];

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
                        memcpy(dstPtrTemp, srcPtr1Temp, x1 * sizeof(T));
                        srcPtr1Temp += x1;
                        srcPtr2Temp += x1;
                        dstPtrTemp += x1;

                        int bufferLength = roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(alpha);
                        __m128 p0, p1, p2, p3;
                        __m128 q0, q1, q2, q3;
                        __m128i px0, px1, px2, px3;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            q0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            q1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            q2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            q3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), q0);    // pixels 0-3
                            p1 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), q1);    // pixels 4-7
                            p2 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), q2);    // pixels 8-11
                            p3 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), q3);    // pixels 12-15

                            p0 = _mm_mul_ps(p0, pMul);
                            p1 = _mm_mul_ps(p1, pMul);
                            p2 = _mm_mul_ps(p2, pMul);
                            p3 = _mm_mul_ps(p3, pMul);
                            px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
                            px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
                            px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
                            px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));

                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtr1Temp +=16;
                            srcPtr2Temp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK((alpha * (*srcPtr1Temp - *srcPtr2Temp)) + *srcPtr2Temp);
                            dstPtrTemp++;
                            srcPtr2Temp++;
                            srcPtr1Temp++;
                        }

                        memcpy(dstPtrTemp, srcPtr1Temp, remainingElementsAfterROI * sizeof(T));
                        srcPtr1Temp += remainingElementsAfterROI;
                        srcPtr2Temp += remainingElementsAfterROI;
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

            Rpp32f alpha = batch_alpha[batchCount];

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
                    memcpy(dstPtrTemp, srcPtr1Temp, elementsBeforeROI * sizeof(T));
                    srcPtr1Temp += elementsBeforeROI;
                    srcPtr2Temp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    int bufferLength = channel * roiPoints[batchCount].roiWidth;
                    int alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(alpha);
                    __m128 p0, p1, p2, p3;
                    __m128 q0, q1, q2, q3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        q0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        q1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        q2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        q3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), q0);    // pixels 0-3
                        p1 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), q1);    // pixels 4-7
                        p2 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), q2);    // pixels 8-11
                        p3 = _mm_sub_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), q3);    // pixels 12-15

                        p0 = _mm_mul_ps(p0, pMul);
                        p1 = _mm_mul_ps(p1, pMul);
                        p2 = _mm_mul_ps(p2, pMul);
                        p3 = _mm_mul_ps(p3, pMul);
                        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
                        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
                        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
                        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK((alpha * (*srcPtr1Temp - *srcPtr2Temp)) + *srcPtr2Temp);
                        dstPtrTemp++;
                        srcPtr2Temp++;
                        srcPtr1Temp++;
                    }

                    memcpy(dstPtrTemp, srcPtr1Temp, remainingElementsAfterROI * sizeof(T));
                    srcPtr1Temp += remainingElementsAfterROI;
                    srcPtr2Temp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus blend_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                        Rpp32f alpha, RppiChnFormat chnFormat,
                        unsigned int channel)
{
    T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;

    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(alpha);
    __m128 p0, p1, p2, p3;
    __m128 q0, q1, q2, q3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);

        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

        px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        q0 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), p0);    // pixels 0-3
        q1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), p1);    // pixels 4-7
        q2 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), p2);    // pixels 8-11
        q3 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), p3);    // pixels 12-15

        q0 = _mm_mul_ps(q0, pMul);
        q1 = _mm_mul_ps(q1, pMul);
        q2 = _mm_mul_ps(q2, pMul);
        q3 = _mm_mul_ps(q3, pMul);
        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));

        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

        srcPtr1Temp +=16;
        srcPtr2Temp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp = (T) RPPPIXELCHECK(*srcPtr1Temp + (alpha * (*srcPtr2Temp - *srcPtr1Temp)));
        dstPtrTemp++;
        srcPtr2Temp++;
        srcPtr1Temp++;
    }

    return RPP_SUCCESS;
}

/************ gamma_correction ************/

template <typename T>
RppStatus gamma_correction_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                      Rpp32f *batch_gamma,
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
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
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

            Rpp32f gamma = batch_gamma[batchCount];

            Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));

            for (int i = 0; i < 256; i++)
            {
                gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
            }

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
                                *dstPtrTemp = gammaLUT[*srcPtrTemp];

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
            free(gammaLUT);
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
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
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

            Rpp32f gamma = batch_gamma[batchCount];

            Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));

            for (int i = 0; i < 256; i++)
            {
                gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
            }

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
                                *dstPtrTemp = gammaLUT[*srcPtrTemp];

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
            free(gammaLUT);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageBuffer = channel * srcSize.height * srcSize.width;

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));

    for (int i = 0; i < 256; i++)
    {
        gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
    }

    for (int i = 0; i < imageBuffer; i++)
    {
        *dstPtrTemp = gammaLUT[*srcPtrTemp];
        srcPtrTemp++;
        dstPtrTemp++;
    }

    free(gammaLUT);

    return RPP_SUCCESS;

}

/************ exposure ************/

template <typename T>
RppStatus exposure_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32f *batch_exposureFactor,
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

            Rpp32f exposureFactor = batch_exposureFactor[batchCount];
            Rpp32f multiplyingFactor = pow(2, exposureFactor);

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
                    Rpp32f pixel;

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
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(multiplyingFactor);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1, px2, px3;

                        int i = 0;
                        for (; i < alignedLength; i+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));

                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; i < bufferLength; i++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
                        }

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

            Rpp32f exposureFactor = batch_exposureFactor[batchCount];
            Rpp32f multiplyingFactor = pow(2, exposureFactor);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

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

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(multiplyingFactor);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
                    }

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus exposure_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f exposureFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f multiplyingFactor = pow(2, exposureFactor);

    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(multiplyingFactor);
    __m128 p0, p1, p2, p3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));

        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

        srcPtrTemp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
    }

    return RPP_SUCCESS;
}

/**************** blur ***************/

template <typename T>
RppStatus blur_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32f multiplier = 1.0 / (kernelSize * kernelSize);
            Rpp32u bound = (kernelSize - 1) / 2;

            Rpp32u firstRow = y1 + bound;
            Rpp32u firstColumn = x1 + bound;
            Rpp32u lastRow = y2 - bound;
            Rpp32u lastColumn = x2 - bound;

            Rpp32u roiWidthToCompute = lastColumn - firstColumn + 1;
            Rpp32u remainingElementsAfterROI = batch_srcSize[batchCount].width - roiWidthToCompute;
            Rpp32u incrementToNextKernel = (kernelSize * batch_srcSizeMax[batchCount].width) - 1;

            Rpp32u sums[25] = {0};
            Rpp32f pixel;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T *srcPtrChannel, *dstPtrChannel;
            T *srcPtrTemp2, *srcPtrTemp3, *srcPtrTemp4;

            if(kernelSize == 5)
            {
                Rpp16u vSums[16] = {0};
                Rpp16u hSums[12] = {0};

                for(int c = 0; c < channel; c++)
                {
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);

                    srcPtrTemp2 = srcPtrChannel + ((firstRow - bound) * batch_srcSizeMax[batchCount].width) + (firstColumn - bound);


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        if (i < firstRow || i > lastRow)
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, firstColumn * sizeof(T));
                            srcPtrTemp += firstColumn;
                            dstPtrTemp += firstColumn;

                            srcPtrTemp3 = srcPtrTemp2;

                            int bufferLength = roiWidthToCompute;
                            int alignedLength = (bufferLength / 12) * 12;

                            __m128i const zero = _mm_setzero_si128();
                            __m128i px0, px1, px2, qx0, qx1;
                            __m128i pSum1, pSum2;
                            __m128 p0, p1, p2;

                            __m128 pMul = _mm_set1_ps(multiplier);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength - 4; vectorLoopCount+=12)
                            {
                                pSum1 = _mm_set1_epi16(0);
                                pSum2 = _mm_set1_epi16(0);

                                srcPtrTemp4 = srcPtrTemp3;

                                for (int m = 0; m < kernelSize; m++)
                                {
                                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp4);

                                    px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                                    px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                                    pSum1 = _mm_add_epi16(px1, pSum1);
                                    pSum2 = _mm_add_epi16(px2, pSum2);

                                    srcPtrTemp4 += batch_srcSizeMax[batchCount].width;
                                }

                                _mm_storeu_si128((__m128i *)vSums, pSum1);
                                _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);

                                hSums[0] = vSums[0] + vSums[1] + vSums[2] + vSums[3] + vSums[4];
                                for (int idxCurr = 1, idxPrev = 0, idxNext = 5; idxCurr < 12; idxCurr++, idxPrev++, idxNext++)
                                {
                                    hSums[idxCurr] = hSums[idxPrev] - vSums[idxPrev] + vSums[idxNext];
                                }

                                qx0 = _mm_loadu_si128((__m128i *)hSums);
                                qx1 = _mm_loadu_si128((__m128i *)(hSums + 8));

                                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx0, zero));    // pixels 0-3
                                p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx0, zero));    // pixels 4-7
                                p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx1, zero));    // pixels 8-11

                                p0 = _mm_mul_ps(p0, pMul);
                                p1 = _mm_mul_ps(p1, pMul);
                                p2 = _mm_mul_ps(p2, pMul);

                                px0 = _mm_cvtps_epi32(p0);
                                px1 = _mm_cvtps_epi32(p1);
                                px2 = _mm_cvtps_epi32(p2);

                                px0 = _mm_packus_epi32(px0, px1);
                                px1 = _mm_packus_epi32(px2, zero);

                                px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                                srcPtrTemp3 = srcPtrTemp3 + 12;
                                srcPtrTemp = srcPtrTemp + 12;
                                dstPtrTemp = dstPtrTemp + 12;
                            }

                            if (vectorLoopCount < bufferLength)
                            {
                                srcPtrTemp4 = srcPtrTemp3;

                                pixel = 0;
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    sums[m] = 0;
                                    for (int n = 0; n < kernelSize; n++)
                                    {
                                        sums[m] += *(srcPtrTemp4 + n);
                                    }
                                    srcPtrTemp4 += batch_srcSizeMax[batchCount].width;
                                }
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    pixel += sums[m];
                                }

                                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                                srcPtrTemp++;
                                vectorLoopCount++;

                                srcPtrTemp4 = srcPtrTemp3 + kernelSize;

                                for(; vectorLoopCount < bufferLength; vectorLoopCount++)
                                {
                                    pixel = 0;
                                    for (int m = 0; m < kernelSize; m++)
                                    {
                                        sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                                        srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                                        srcPtrTemp4 += batch_srcSizeMax[batchCount].width;
                                    }
                                    for (int m = 0; m < kernelSize; m++)
                                    {
                                        pixel += sums[m];
                                    }

                                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                                    srcPtrTemp++;
                                    srcPtrTemp3 -= incrementToNextKernel;
                                    srcPtrTemp4 -= incrementToNextKernel;
                                }
                            }
                            srcPtrTemp2 += batch_srcSizeMax[batchCount].width;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        }
                    }
                }
            }
            else
            {
                for(int c = 0; c < channel; c++)
                {
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);

                    srcPtrTemp2 = srcPtrChannel + ((firstRow - bound) * batch_srcSizeMax[batchCount].width) + (firstColumn - bound);


                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                        if (i < firstRow || i > lastRow)
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, firstColumn * sizeof(T));
                            srcPtrTemp += firstColumn;
                            dstPtrTemp += firstColumn;

                            srcPtrTemp3 = srcPtrTemp2;

                            pixel = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sums[m] = 0;
                                for (int n = 0; n < kernelSize; n++)
                                {
                                    sums[m] += *(srcPtrTemp3 + n);
                                }
                                srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixel += sums[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                            srcPtrTemp++;

                            srcPtrTemp3 = srcPtrTemp2;
                            srcPtrTemp4 = srcPtrTemp2 + kernelSize;

                            for (int j= 1; j < roiWidthToCompute; j++)
                            {
                                pixel = 0;
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                                    srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                                    srcPtrTemp4 += batch_srcSizeMax[batchCount].width;
                                }
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    pixel += sums[m];
                                }

                                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                                srcPtrTemp++;
                                srcPtrTemp3 -= incrementToNextKernel;
                                srcPtrTemp4 -= incrementToNextKernel;
                            }
                            srcPtrTemp2 += batch_srcSizeMax[batchCount].width;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32f multiplier = 1.0 / (kernelSize * kernelSize);
            Rpp32u bound = (kernelSize - 1) / 2;

            Rpp32u firstRow = y1 + bound;
            Rpp32u firstColumn = x1 + bound;
            Rpp32u lastRow = y2 - bound;
            Rpp32u lastColumn = x2 - bound;

            Rpp32u roiWidthToCompute = lastColumn - firstColumn + 1;

            Rpp32u sumsR[25] = {0}, sumsG[25] = {0}, sumsB[25] = {0};
            Rpp32f pixelR, pixelG, pixelB;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInKernelRow = channel * kernelSize;
            Rpp32u incrementToNextKernel = (kernelSize * elementsInRowMax) - channel;
            Rpp32u channeledBound = channel * bound;
            Rpp32u channeledFirstColumn = channel * firstColumn;

            T *srcPtrTemp2R, *srcPtrTemp2G, *srcPtrTemp2B;
            T *srcPtrTemp3R, *srcPtrTemp3G, *srcPtrTemp3B;
            T *srcPtrTemp4R, *srcPtrTemp4G, *srcPtrTemp4B;

            srcPtrTemp2R = srcPtrImage + ((firstRow - bound) * elementsInRowMax) + ((firstColumn - bound) * channel);
            srcPtrTemp2G = srcPtrTemp2R + 1;
            srcPtrTemp2B = srcPtrTemp2R + 2;

            if (kernelSize == 5)
            {
                Rpp16u vSums[32] = {0};
                Rpp16u hSums[15] = {0};

                T *srcPtrTemp3a, *srcPtrTemp3b;
                T *srcPtrTemp4a, *srcPtrTemp4b;


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (i < firstRow || i > lastRow)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn * sizeof(T));
                        dstPtrTemp += channeledFirstColumn;
                        srcPtrTemp += channeledFirstColumn;

                        srcPtrTemp3a = srcPtrTemp2R;
                        srcPtrTemp3b = srcPtrTemp3a + 16;

                        int bufferLength = roiWidthToCompute * channel;
                        int alignedLength = (bufferLength / 15) * 15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0a, px1a, px2a;
                        __m128i px0b, px1b, px2b;
                        __m128i qx0, qx1;
                        __m128 p0, p1, p2, p3;
                        __m128i pSum1a, pSum2a, pSum1b, pSum2b;

                        __m128 pMul = _mm_set1_ps(multiplier);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength - 17; vectorLoopCount+=15)
                        {
                            pSum1a = _mm_set1_epi16(0);
                            pSum2a = _mm_set1_epi16(0);
                            pSum1b = _mm_set1_epi16(0);
                            pSum2b = _mm_set1_epi16(0);

                            srcPtrTemp4a = srcPtrTemp3a;
                            srcPtrTemp4b = srcPtrTemp3b;

                            for (int m = 0; m < kernelSize; m++)
                            {
                                px0a =  _mm_loadu_si128((__m128i *)srcPtrTemp4a);
                                px0b =  _mm_loadu_si128((__m128i *)srcPtrTemp4b);

                                px1a = _mm_unpacklo_epi8(px0a, zero);    // pixels 0-7
                                px2a = _mm_unpackhi_epi8(px0a, zero);    // pixels 8-15
                                px1b = _mm_unpacklo_epi8(px0b, zero);    // pixels 16-23
                                px2b = _mm_unpackhi_epi8(px0b, zero);    // pixels 24-31

                                pSum1a = _mm_add_epi16(px1a, pSum1a);
                                pSum2a = _mm_add_epi16(px2a, pSum2a);
                                pSum1b = _mm_add_epi16(px1b, pSum1b);
                                pSum2b = _mm_add_epi16(px2b, pSum2b);

                                srcPtrTemp4a += elementsInRowMax;
                                srcPtrTemp4b += elementsInRowMax;
                            }

                            _mm_storeu_si128((__m128i *)vSums, pSum1a);
                            _mm_storeu_si128((__m128i *)(vSums + 8), pSum2a);
                            _mm_storeu_si128((__m128i *)(vSums + 16), pSum1b);
                            _mm_storeu_si128((__m128i *)(vSums + 24), pSum2b);

                            hSums[0] = vSums[0] + vSums[3] + vSums[6] + vSums[9] + vSums[12];
                            hSums[1] = vSums[1] + vSums[4] + vSums[7] + vSums[10] + vSums[13];
                            hSums[2] = vSums[2] + vSums[5] + vSums[8] + vSums[11] + vSums[14];
                            for (int idxCurr = 3, idxPrev = 0, idxNext = 15; idxCurr < 15; idxCurr++, idxPrev++, idxNext++)
                            {
                                hSums[idxCurr] = hSums[idxPrev] - vSums[idxPrev] + vSums[idxNext];
                            }

                            qx0 = _mm_loadu_si128((__m128i *)hSums);
                            qx1 = _mm_loadu_si128((__m128i *)(hSums + 8));

                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx1, zero));    // pixels 12-15

                            p0 = _mm_mul_ps(p0, pMul);
                            p1 = _mm_mul_ps(p1, pMul);
                            p2 = _mm_mul_ps(p2, pMul);
                            p3 = _mm_mul_ps(p3, pMul);

                            px0a = _mm_cvtps_epi32(p0);
                            px1a = _mm_cvtps_epi32(p1);
                            px0b = _mm_cvtps_epi32(p2);
                            px1b = _mm_cvtps_epi32(p3);

                            px0a = _mm_packus_epi32(px0a, px1a);
                            px0b = _mm_packus_epi32(px0b, px1b);

                            px0a = _mm_packus_epi16(px0a, px0b);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0a);

                            srcPtrTemp3a = srcPtrTemp3a + 15;
                            srcPtrTemp3b = srcPtrTemp3b + 15;
                            srcPtrTemp = srcPtrTemp + 15;
                            dstPtrTemp = dstPtrTemp + 15;
                        }
                        Rpp32u remainingPixels = bufferLength - vectorLoopCount + 1;

                        srcPtrTemp2R += elementsInRowMax;
                        srcPtrTemp2G += elementsInRowMax;
                        srcPtrTemp2B += elementsInRowMax;

                        memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn + remainingPixels * sizeof(T));
                        dstPtrTemp += channeledFirstColumn + remainingPixels;
                        srcPtrTemp += channeledFirstColumn + remainingPixels;
                    }
                }
            }
            else
            {

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (i < firstRow || i > lastRow)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn * sizeof(T));
                        dstPtrTemp += channeledFirstColumn;
                        srcPtrTemp += channeledFirstColumn;

                        srcPtrTemp3R = srcPtrTemp2R;
                        srcPtrTemp3G = srcPtrTemp2G;
                        srcPtrTemp3B = srcPtrTemp2B;

                        pixelR = 0;
                        pixelG = 0;
                        pixelB = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sumsR[m] = 0;
                            sumsG[m] = 0;
                            sumsB[m] = 0;
                            for (int n = 0; n < elementsInKernelRow; n += channel)
                            {
                                sumsR[m] += *(srcPtrTemp3R + n);
                                sumsG[m] += *(srcPtrTemp3G + n);
                                sumsB[m] += *(srcPtrTemp3B + n);
                            }
                            srcPtrTemp3R += elementsInRowMax;
                            srcPtrTemp3G += elementsInRowMax;
                            srcPtrTemp3B += elementsInRowMax;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixelR += sumsR[m];
                            pixelG += sumsG[m];
                            pixelB += sumsB[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                        srcPtrTemp += channel;

                        srcPtrTemp3R = srcPtrTemp2R;
                        srcPtrTemp3G = srcPtrTemp2G;
                        srcPtrTemp3B = srcPtrTemp2B;
                        srcPtrTemp4R = srcPtrTemp2R + elementsInKernelRow;
                        srcPtrTemp4G = srcPtrTemp2G + elementsInKernelRow;
                        srcPtrTemp4B = srcPtrTemp2B + elementsInKernelRow;

                        for (int j= 1; j < roiWidthToCompute; j++)
                        {
                            pixelR = 0;
                            pixelG = 0;
                            pixelB = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sumsR[m] = sumsR[m] - (Rpp32f) *srcPtrTemp3R + (Rpp32f) *srcPtrTemp4R;
                                sumsG[m] = sumsG[m] - (Rpp32f) *srcPtrTemp3G + (Rpp32f) *srcPtrTemp4G;
                                sumsB[m] = sumsB[m] - (Rpp32f) *srcPtrTemp3B + (Rpp32f) *srcPtrTemp4B;
                                srcPtrTemp3R += elementsInRowMax;
                                srcPtrTemp3G += elementsInRowMax;
                                srcPtrTemp3B += elementsInRowMax;
                                srcPtrTemp4R += elementsInRowMax;
                                srcPtrTemp4G += elementsInRowMax;
                                srcPtrTemp4B += elementsInRowMax;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixelR += sumsR[m];
                                pixelG += sumsG[m];
                                pixelB += sumsB[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                            srcPtrTemp += channel;
                            srcPtrTemp3R -= incrementToNextKernel;
                            srcPtrTemp3G -= incrementToNextKernel;
                            srcPtrTemp3B -= incrementToNextKernel;
                            srcPtrTemp4R -= incrementToNextKernel;
                            srcPtrTemp4G -= incrementToNextKernel;
                            srcPtrTemp4B -= incrementToNextKernel;
                        }

                        srcPtrTemp2R += elementsInRowMax;
                        srcPtrTemp2G += elementsInRowMax;
                        srcPtrTemp2B += elementsInRowMax;

                        memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn * sizeof(T));
                        dstPtrTemp += channeledFirstColumn;
                        srcPtrTemp += channeledFirstColumn;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f multiplier = 1.0 / (kernelSize * kernelSize);

    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u bound = (kernelSize - 1) / 2;

    Rpp32u lastRow = srcSize.height - 1 - bound;
    Rpp32u lastColumn = srcSize.width - 1 - bound;

    Rpp32u widthToCompute = srcSize.width - (2 * bound);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTemp, *srcPtrTemp2, *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;

        Rpp32u sums[25] = {0};
        Rpp32f pixel;

        Rpp32u incrementToNextKernel = (kernelSize * srcSize.width)  - 1;

        if (kernelSize == 5)
        {
            Rpp16u vSums[16] = {0};
            Rpp16u hSums[12] = {0};

            for (int c = 0; c < channel; c++)
            {
                T *srcPtrTemp = srcPtr + (c * imageDim);
                T *dstPtrTemp = dstPtr + (c * imageDim);

                srcPtrTemp2 = srcPtrTemp;

                for (int i = 0; i < srcSize.height; i++)
                {
                    if (i < bound || i > lastRow)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));
                        srcPtrTemp += srcSize.width;
                        dstPtrTemp += srcSize.width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                        dstPtrTemp += bound;
                        srcPtrTemp += bound;

                        srcPtrTemp3 = srcPtrTemp2;

                        int bufferLength = widthToCompute;
                        int alignedLength = (bufferLength / 12) * 12;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, qx0, qx1;
                        __m128i pSum1, pSum2;
                        __m128 p0, p1, p2;

                        __m128 pMul = _mm_set1_ps(multiplier);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength - 4; vectorLoopCount+=12)
                        {
                            pSum1 = _mm_set1_epi16(0);
                            pSum2 = _mm_set1_epi16(0);

                            srcPtrTemp4 = srcPtrTemp3;

                            for (int m = 0; m < kernelSize; m++)
                            {
                                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp4);

                                px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                                px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                                pSum1 = _mm_add_epi16(px1, pSum1);
                                pSum2 = _mm_add_epi16(px2, pSum2);

                                srcPtrTemp4 += srcSize.width;
                            }

                            _mm_storeu_si128((__m128i *)vSums, pSum1);
                            _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);

                            hSums[0] = vSums[0] + vSums[1] + vSums[2] + vSums[3] + vSums[4];
                            for (int idxCurr = 1, idxPrev = 0, idxNext = 5; idxCurr < 12; idxCurr++, idxPrev++, idxNext++)
                            {
                                hSums[idxCurr] = hSums[idxPrev] - vSums[idxPrev] + vSums[idxNext];
                            }

                            qx0 = _mm_loadu_si128((__m128i *)hSums);
                            qx1 = _mm_loadu_si128((__m128i *)(hSums + 8));

                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx1, zero));    // pixels 8-11

                            p0 = _mm_mul_ps(p0, pMul);
                            p1 = _mm_mul_ps(p1, pMul);
                            p2 = _mm_mul_ps(p2, pMul);

                            px0 = _mm_cvtps_epi32(p0);
                            px1 = _mm_cvtps_epi32(p1);
                            px2 = _mm_cvtps_epi32(p2);

                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, zero);

                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp3 = srcPtrTemp3 + 12;
                            srcPtrTemp = srcPtrTemp + 12;
                            dstPtrTemp = dstPtrTemp + 12;
                        }

                        if (vectorLoopCount < bufferLength)
                        {
                            srcPtrTemp4 = srcPtrTemp3;

                            pixel = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sums[m] = 0;
                                for (int n = 0; n < kernelSize; n++)
                                {
                                    sums[m] += *(srcPtrTemp4 + n);
                                }
                                srcPtrTemp4 += srcSize.width;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixel += sums[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                            srcPtrTemp++;
                            vectorLoopCount++;

                            srcPtrTemp4 = srcPtrTemp3 + kernelSize;

                            for(; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                pixel = 0;
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                                    srcPtrTemp3 += srcSize.width;
                                    srcPtrTemp4 += srcSize.width;
                                }
                                for (int m = 0; m < kernelSize; m++)
                                {
                                    pixel += sums[m];
                                }

                                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                                srcPtrTemp++;
                                srcPtrTemp3 -= incrementToNextKernel;
                                srcPtrTemp4 -= incrementToNextKernel;
                            }
                        }
                        srcPtrTemp2 += srcSize.width;

                        memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                        dstPtrTemp += bound;
                        srcPtrTemp += bound;
                    }
                }
            }
        }
        else
        {
            for (int c = 0; c < channel; c++)
            {
                T *srcPtrTemp = srcPtr + (c * imageDim);
                T *dstPtrTemp = dstPtr + (c * imageDim);

                srcPtrTemp2 = srcPtrTemp;

                for (int i = 0; i < srcSize.height; i++)
                {
                    if (i < bound || i > lastRow)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));
                        srcPtrTemp += srcSize.width;
                        dstPtrTemp += srcSize.width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                        dstPtrTemp += bound;
                        srcPtrTemp += bound;

                        srcPtrTemp3 = srcPtrTemp2;

                        pixel = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sums[m] = 0;
                            for (int n = 0; n < kernelSize; n++)
                            {
                                sums[m] += *(srcPtrTemp3 + n);
                            }
                            srcPtrTemp3 += srcSize.width;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixel += sums[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                        srcPtrTemp++;

                        srcPtrTemp3 = srcPtrTemp2;
                        srcPtrTemp4 = srcPtrTemp2 + kernelSize;

                        for (int j= 1; j < widthToCompute; j++)
                        {
                            pixel = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                                srcPtrTemp3 += srcSize.width;
                                srcPtrTemp4 += srcSize.width;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixel += sums[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel * multiplier);
                            srcPtrTemp++;
                            srcPtrTemp3 -= incrementToNextKernel;
                            srcPtrTemp4 -= incrementToNextKernel;
                        }
                        srcPtrTemp2 += srcSize.width;

                        memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                        dstPtrTemp += bound;
                        srcPtrTemp += bound;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTemp, *dstPtrTemp;
        T *srcPtrTemp2R, *srcPtrTemp2G, *srcPtrTemp2B;
        T *srcPtrTemp3R, *srcPtrTemp3G, *srcPtrTemp3B;
        T *srcPtrTemp4R, *srcPtrTemp4G, *srcPtrTemp4B;

        Rpp32u sumsR[25] = {0}, sumsG[25] = {0}, sumsB[25] = {0};
        Rpp32f pixelR, pixelG, pixelB;

        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u elementsInKernelRow = channel * kernelSize;
        Rpp32u incrementToNextKernel = (kernelSize * elementsInRow) - channel;
        Rpp32u channeledBound = channel * bound;

        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;
        srcPtrTemp2R = srcPtrTemp;
        srcPtrTemp2G = srcPtrTemp + 1;
        srcPtrTemp2B = srcPtrTemp + 2;

        if (kernelSize == 5)
        {
            Rpp16u vSums[32] = {0};
            Rpp16u hSums[15] = {0};

            T *srcPtrTemp3a, *srcPtrTemp3b;
            T *srcPtrTemp4a, *srcPtrTemp4b;

            for (int i = 0; i < srcSize.height; i++)
            {
                if (i < bound || i > lastRow)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                    srcPtrTemp += elementsInRow;
                    dstPtrTemp += elementsInRow;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                    dstPtrTemp += channeledBound;
                    srcPtrTemp += channeledBound;

                    srcPtrTemp3a = srcPtrTemp2R;
                    srcPtrTemp3b = srcPtrTemp3a + 16;

                    int bufferLength = widthToCompute * channel;
                    int alignedLength = (bufferLength / 15) * 15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0a, px1a, px2a;
                    __m128i px0b, px1b, px2b;
                    __m128i qx0, qx1;
                    __m128 p0, p1, p2, p3;
                    __m128i pSum1a, pSum2a, pSum1b, pSum2b;

                    __m128 pMul = _mm_set1_ps(multiplier);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength - 17; vectorLoopCount+=15)
                    {
                        pSum1a = _mm_set1_epi16(0);
                        pSum2a = _mm_set1_epi16(0);
                        pSum1b = _mm_set1_epi16(0);
                        pSum2b = _mm_set1_epi16(0);

                        srcPtrTemp4a = srcPtrTemp3a;
                        srcPtrTemp4b = srcPtrTemp3b;

                        for (int m = 0; m < kernelSize; m++)
                        {
                            px0a =  _mm_loadu_si128((__m128i *)srcPtrTemp4a);
                            px0b =  _mm_loadu_si128((__m128i *)srcPtrTemp4b);

                            px1a = _mm_unpacklo_epi8(px0a, zero);    // pixels 0-7
                            px2a = _mm_unpackhi_epi8(px0a, zero);    // pixels 8-15
                            px1b = _mm_unpacklo_epi8(px0b, zero);    // pixels 16-23
                            px2b = _mm_unpackhi_epi8(px0b, zero);    // pixels 24-31

                            pSum1a = _mm_add_epi16(px1a, pSum1a);
                            pSum2a = _mm_add_epi16(px2a, pSum2a);
                            pSum1b = _mm_add_epi16(px1b, pSum1b);
                            pSum2b = _mm_add_epi16(px2b, pSum2b);

                            srcPtrTemp4a += elementsInRow;
                            srcPtrTemp4b += elementsInRow;
                        }

                        _mm_storeu_si128((__m128i *)vSums, pSum1a);
                        _mm_storeu_si128((__m128i *)(vSums + 8), pSum2a);
                        _mm_storeu_si128((__m128i *)(vSums + 16), pSum1b);
                        _mm_storeu_si128((__m128i *)(vSums + 24), pSum2b);

                        hSums[0] = vSums[0] + vSums[3] + vSums[6] + vSums[9] + vSums[12];
                        hSums[1] = vSums[1] + vSums[4] + vSums[7] + vSums[10] + vSums[13];
                        hSums[2] = vSums[2] + vSums[5] + vSums[8] + vSums[11] + vSums[14];
                        for (int idxCurr = 3, idxPrev = 0, idxNext = 15; idxCurr < 15; idxCurr++, idxPrev++, idxNext++)
                        {
                            hSums[idxCurr] = hSums[idxPrev] - vSums[idxPrev] + vSums[idxNext];
                        }

                        qx0 = _mm_loadu_si128((__m128i *)hSums);
                        qx1 = _mm_loadu_si128((__m128i *)(hSums + 8));

                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(qx1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(qx1, zero));    // pixels 12-15

                        p0 = _mm_mul_ps(p0, pMul);
                        p1 = _mm_mul_ps(p1, pMul);
                        p2 = _mm_mul_ps(p2, pMul);
                        p3 = _mm_mul_ps(p3, pMul);

                        px0a = _mm_cvtps_epi32(p0);
                        px1a = _mm_cvtps_epi32(p1);
                        px0b = _mm_cvtps_epi32(p2);
                        px1b = _mm_cvtps_epi32(p3);

                        px0a = _mm_packus_epi32(px0a, px1a);
                        px0b = _mm_packus_epi32(px0b, px1b);

                        px0a = _mm_packus_epi16(px0a, px0b);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0a);

                        srcPtrTemp3a = srcPtrTemp3a + 15;
                        srcPtrTemp3b = srcPtrTemp3b + 15;
                        srcPtrTemp = srcPtrTemp + 15;
                        dstPtrTemp = dstPtrTemp + 15;
                    }

                    if (vectorLoopCount < bufferLength)
                    {
                        srcPtrTemp3R = srcPtrTemp3a;
                        srcPtrTemp3G = srcPtrTemp3R + 1;
                        srcPtrTemp3B = srcPtrTemp3R + 2;

                        srcPtrTemp4R = srcPtrTemp3R;
                        srcPtrTemp4G = srcPtrTemp3G;
                        srcPtrTemp4B = srcPtrTemp3B;

                        pixelR = 0;
                        pixelG = 0;
                        pixelB = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sumsR[m] = 0;
                            sumsG[m] = 0;
                            sumsB[m] = 0;
                            for (int n = 0; n < elementsInKernelRow; n += channel)
                            {
                                sumsR[m] += *(srcPtrTemp4R + n);
                                sumsG[m] += *(srcPtrTemp4G + n);
                                sumsB[m] += *(srcPtrTemp4B + n);
                            }
                            srcPtrTemp4R += elementsInRow;
                            srcPtrTemp4G += elementsInRow;
                            srcPtrTemp4B += elementsInRow;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixelR += sumsR[m];
                            pixelG += sumsG[m];
                            pixelB += sumsB[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                        srcPtrTemp += channel;
                        vectorLoopCount += channel;

                        srcPtrTemp4R = srcPtrTemp3R + elementsInKernelRow;
                        srcPtrTemp4G = srcPtrTemp3G + elementsInKernelRow;
                        srcPtrTemp4B = srcPtrTemp3B + elementsInKernelRow;

                        for(; vectorLoopCount < bufferLength; vectorLoopCount += channel)
                        {
                            pixelR = 0;
                            pixelG = 0;
                            pixelB = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sumsR[m] = sumsR[m] - (Rpp32f) *srcPtrTemp3R + (Rpp32f) *srcPtrTemp4R;
                                sumsG[m] = sumsG[m] - (Rpp32f) *srcPtrTemp3G + (Rpp32f) *srcPtrTemp4G;
                                sumsB[m] = sumsB[m] - (Rpp32f) *srcPtrTemp3B + (Rpp32f) *srcPtrTemp4B;
                                srcPtrTemp3R += elementsInRow;
                                srcPtrTemp3G += elementsInRow;
                                srcPtrTemp3B += elementsInRow;
                                srcPtrTemp4R += elementsInRow;
                                srcPtrTemp4G += elementsInRow;
                                srcPtrTemp4B += elementsInRow;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixelR += sumsR[m];
                                pixelG += sumsG[m];
                                pixelB += sumsB[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                            srcPtrTemp += channel;
                            srcPtrTemp3R -= incrementToNextKernel;
                            srcPtrTemp3G -= incrementToNextKernel;
                            srcPtrTemp3B -= incrementToNextKernel;
                            srcPtrTemp4R -= incrementToNextKernel;
                            srcPtrTemp4G -= incrementToNextKernel;
                            srcPtrTemp4B -= incrementToNextKernel;
                        }
                    }

                    srcPtrTemp2R += elementsInRow;

                    memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                    dstPtrTemp += channeledBound;
                    srcPtrTemp += channeledBound;
                }
            }
        }
        else
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                if (i < bound || i > lastRow)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                    srcPtrTemp += elementsInRow;
                    dstPtrTemp += elementsInRow;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                    dstPtrTemp += channeledBound;
                    srcPtrTemp += channeledBound;

                    srcPtrTemp3R = srcPtrTemp2R;
                    srcPtrTemp3G = srcPtrTemp2G;
                    srcPtrTemp3B = srcPtrTemp2B;

                    pixelR = 0;
                    pixelG = 0;
                    pixelB = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        sumsR[m] = 0;
                        sumsG[m] = 0;
                        sumsB[m] = 0;
                        for (int n = 0; n < elementsInKernelRow; n += channel)
                        {
                            sumsR[m] += *(srcPtrTemp3R + n);
                            sumsG[m] += *(srcPtrTemp3G + n);
                            sumsB[m] += *(srcPtrTemp3B + n);
                        }
                        srcPtrTemp3R += elementsInRow;
                        srcPtrTemp3G += elementsInRow;
                        srcPtrTemp3B += elementsInRow;
                    }
                    for (int m = 0; m < kernelSize; m++)
                    {
                        pixelR += sumsR[m];
                        pixelG += sumsG[m];
                        pixelB += sumsB[m];
                    }

                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                    srcPtrTemp += channel;

                    srcPtrTemp3R = srcPtrTemp2R;
                    srcPtrTemp3G = srcPtrTemp2G;
                    srcPtrTemp3B = srcPtrTemp2B;
                    srcPtrTemp4R = srcPtrTemp2R + elementsInKernelRow;
                    srcPtrTemp4G = srcPtrTemp2G + elementsInKernelRow;
                    srcPtrTemp4B = srcPtrTemp2B + elementsInKernelRow;

                    for (int j= 1; j < widthToCompute; j++)
                    {
                        pixelR = 0;
                        pixelG = 0;
                        pixelB = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sumsR[m] = sumsR[m] - (Rpp32f) *srcPtrTemp3R + (Rpp32f) *srcPtrTemp4R;
                            sumsG[m] = sumsG[m] - (Rpp32f) *srcPtrTemp3G + (Rpp32f) *srcPtrTemp4G;
                            sumsB[m] = sumsB[m] - (Rpp32f) *srcPtrTemp3B + (Rpp32f) *srcPtrTemp4B;
                            srcPtrTemp3R += elementsInRow;
                            srcPtrTemp3G += elementsInRow;
                            srcPtrTemp3B += elementsInRow;
                            srcPtrTemp4R += elementsInRow;
                            srcPtrTemp4G += elementsInRow;
                            srcPtrTemp4B += elementsInRow;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixelR += sumsR[m];
                            pixelG += sumsG[m];
                            pixelB += sumsB[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG * multiplier);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB * multiplier);
                        srcPtrTemp += channel;
                        srcPtrTemp3R -= incrementToNextKernel;
                        srcPtrTemp3G -= incrementToNextKernel;
                        srcPtrTemp3B -= incrementToNextKernel;
                        srcPtrTemp4R -= incrementToNextKernel;
                        srcPtrTemp4G -= incrementToNextKernel;
                        srcPtrTemp4B -= incrementToNextKernel;
                    }
                    srcPtrTemp2R += elementsInRow;
                    srcPtrTemp2G += elementsInRow;
                    srcPtrTemp2B += elementsInRow;

                    memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                    dstPtrTemp += channeledBound;
                    srcPtrTemp += channeledBound;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** histogram_balance ***************/

template <typename T>
RppStatus histogram_balance_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
RppStatus histogram_balance_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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

/**************** random_crop_letterbox ***************/

template <typename T>
RppStatus random_crop_letterbox_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                           Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2, RppiROI *roiPoints,
                                           Rpp32u nbatchSize,
                                           RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u x1 = batch_x1[batchCount];
        Rpp32u y1 = batch_y1[batchCount];
        Rpp32u x2 = batch_x2[batchCount];
        Rpp32u y2 = batch_y2[batchCount];

        Rpp32u srcLoc = 0, dstLoc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
        compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);

        T *srcPtrImage = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtrImage = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(srcPtr + srcLoc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImage,
                                            chnFormat, channel);

        Rpp32u borderWidth = (4 * RPPMIN2(batch_dstSize[batchCount].height, batch_dstSize[batchCount].width) / 100);

        RppiSize srcSizeSubImage;
        T* srcPtrSubImage;
        compute_subimage_location_host(srcPtrImage, &srcPtrSubImage, batch_srcSize[batchCount], &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

        RppiSize srcSizeSubImagePadded;
        srcSizeSubImagePadded.height = srcSizeSubImage.height + (2 * borderWidth);
        srcSizeSubImagePadded.width = srcSizeSubImage.width + (2 * borderWidth);

        T *srcPtrImageCrop = (T *)calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));
        generate_crop_host(srcPtrImage, batch_srcSize[batchCount], srcPtrSubImage, srcSizeSubImage, srcPtrImageCrop, chnFormat, channel);

        T *srcPtrImageCropPadded = (T *)calloc(channel * srcSizeSubImagePadded.height * srcSizeSubImagePadded.width, sizeof(T));
        generate_evenly_padded_image_host(srcPtrImageCrop, srcSizeSubImage, srcPtrImageCropPadded, srcSizeSubImagePadded, chnFormat, channel);

        resize_kernel_host(srcPtrImageCropPadded, srcSizeSubImagePadded, dstPtrImage, batch_dstSize[batchCount], chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtr + dstLoc,
                                          chnFormat, channel);

        free(srcPtrImage);
        free(dstPtrImage);
        free(srcPtrImageCrop);
        free(srcPtrImageCropPadded);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus random_crop_letterbox_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                     Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if ((RPPINRANGE(x1, 0, srcSize.width - 1) == 0)
        || (RPPINRANGE(x2, 0, srcSize.width - 1) == 0)
        || (RPPINRANGE(y1, 0, srcSize.height - 1) == 0)
        || (RPPINRANGE(y2, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    Rpp32u borderWidth = (5 * RPPMIN2(dstSize.height, dstSize.width) / 100);

    RppiSize srcSizeSubImage;
    T* srcPtrSubImage;
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    RppiSize srcSizeSubImagePadded;
    srcSizeSubImagePadded.height = srcSizeSubImage.height + (2 * borderWidth);
    srcSizeSubImagePadded.width = srcSizeSubImage.width + (2 * borderWidth);

    T *srcPtrCrop = (T *)calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));
    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrCrop, chnFormat, channel);

    T *srcPtrCropPadded = (T *)calloc(channel * srcSizeSubImagePadded.height * srcSizeSubImagePadded.width, sizeof(T));
    generate_evenly_padded_image_host(srcPtrCrop, srcSizeSubImage, srcPtrCropPadded, srcSizeSubImagePadded, chnFormat, channel);

    resize_kernel_host(srcPtrCropPadded, srcSizeSubImagePadded, dstPtr, dstSize, chnFormat, channel);

    free(srcPtrCrop);
    free(srcPtrCropPadded);

    return RPP_SUCCESS;
}



/**************** Pixelate ***************/

template <typename T>
RppStatus pixelate_base_pln_host(T* srcPtrTemp, Rpp32u elementsInRow, T* dstPtrTemp,
                                 int kernelRowMax, int kernelColMax, int i, int j, Rpp32f multiplier)
{
    Rpp32u sum = 0;

    for(int m = 0 ; m < kernelRowMax ; m++)
    {
        for(int n = 0 ; n < kernelColMax ; n++)
        {
            sum += *(srcPtrTemp + (i + m) * elementsInRow + (j + n));
        }
    }
    sum = RPPPIXELCHECK(sum * multiplier);

    for(int m = 0 ; m < kernelRowMax ; m++)
    {
        for(int n = 0 ; n < kernelColMax ; n++)
        {
            *(dstPtrTemp + (i + m) * elementsInRow + (j + n)) = (T) sum;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus pixelate_base_pkd_host(T* srcPtrTemp, Rpp32u elementsInRow, T* dstPtrTemp,
                                 int kernelRowMax, int kernelColMax, int i, int j, Rpp32f multiplier)
{
    Rpp32u sumR = 0;
    Rpp32u sumG = 0;
    Rpp32u sumB = 0;

    T *loc;

    for(int m = 0 ; m < kernelRowMax ; m++)
    {
        for(int n = 0 ; n < kernelColMax ; n++)
        {
            loc = srcPtrTemp + (i + m) * elementsInRow + ((j + n) * 3);
            sumR += *loc;
            sumG += *(loc + 1);
            sumB += *(loc + 2);
        }
    }
    sumR = RPPPIXELCHECK(sumR * multiplier);
    sumG = RPPPIXELCHECK(sumG * multiplier);
    sumB = RPPPIXELCHECK(sumB * multiplier);

    for(int m = 0 ; m < kernelRowMax ; m++)
    {
        for(int n = 0 ; n < kernelColMax ; n++)
        {
            loc = dstPtrTemp + (i + m) * elementsInRow + ((j + n) * 3);
            *loc = (T) sumR;
            *(loc + 1) = (T) sumG;
            *(loc + 2) = (T) sumB;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus pixelate_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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
            Rpp32u roiEndLoc = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u alignedHeight = (roiPoints[batchCount].roiHeight / 7) * 7;
            Rpp32u heightDiff = roiPoints[batchCount].roiHeight - alignedHeight;

            Rpp32f multiplier = 1.0 / 49;

            Rpp32u increment = batch_srcSizeMax[batchCount].width * 7;
            Rpp32u lastIncrement = batch_srcSizeMax[batchCount].width * heightDiff;
            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 14) * 14;
            if (alignedLength + 2 > bufferLength)
                alignedLength -= 14;
            Rpp32u lengthDiff = bufferLength - alignedLength;
            Rpp32u lengthDiffNew = lengthDiff;
            if (lengthDiff > 7)
                lengthDiffNew -= 7;

            Rpp32f lastColMultiplier = 1.0 / (lengthDiffNew * 7);
            Rpp32f lastRowMultiplier = 1.0 / (heightDiff * 7);
            Rpp32f lastMultiplier = 1.0 / (heightDiff * lengthDiffNew);

            Rpp16u vSums[16] = {0};
            Rpp16u hSums[2] = {0};

            __m128i const zero = _mm_setzero_si128();
            __m128i px0, px1, px2;
            __m128i pSum1, pSum2;


            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                T *srcPtrTemp, *dstPtrTemp, *srcPtrTemp2, *dstPtrTemp2, *srcPtrTemp3, *dstPtrTemp3;
                srcPtrTemp = srcPtrChannel;
                dstPtrTemp = dstPtrChannel;

                for (int i = 0; i < y1; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }

                int i = 0;
                for (; i < alignedHeight; i += 7)
                {
                    srcPtrTemp2 = srcPtrTemp;
                    dstPtrTemp2 = dstPtrTemp;

                    srcPtrTemp3 = srcPtrTemp + roiEndLoc;
                    dstPtrTemp3 = dstPtrTemp + roiEndLoc;

                    for (int i2 = 0; i2 < 7; i2++)
                    {
                        memcpy(dstPtrTemp2, srcPtrTemp2, x1 * sizeof(T));
                        memcpy(dstPtrTemp3, srcPtrTemp3, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp2 += batch_srcSizeMax[batchCount].width;
                        dstPtrTemp2 += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                        dstPtrTemp3 += batch_srcSizeMax[batchCount].width;
                    }

                    srcPtrTemp2 = srcPtrTemp + x1;
                    dstPtrTemp2 = dstPtrTemp + x1;

                    Rpp32u vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=14)
                    {
                        srcPtrTemp3 = srcPtrTemp2;
                        dstPtrTemp3 = dstPtrTemp2;

                        pSum1 = _mm_set1_epi16(0);
                        pSum2 = _mm_set1_epi16(0);

                        for (int m = 0; m < 7; m++)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp3);

                            px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                            pSum1 = _mm_add_epi16(px1, pSum1);
                            pSum2 = _mm_add_epi16(px2, pSum2);

                            srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                        }

                        _mm_storeu_si128((__m128i *)vSums, pSum1);
                        _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);

                        hSums[0] = vSums[0] + vSums[1] + vSums[2] + vSums[3] + vSums[4] + vSums[5] + vSums[6];
                        hSums[1] = vSums[7] + vSums[8] + vSums[9] + vSums[10] + vSums[11] + vSums[12] + vSums[13];

                        px1 = _mm_set1_epi16((Rpp16s) RPPPIXELCHECK((Rpp32f) hSums[0] * multiplier));
                        px2 = _mm_set1_epi16((Rpp16s) RPPPIXELCHECK((Rpp32f) hSums[1] * multiplier));

                        px0 = _mm_packus_epi16(px1, px2);

                        for (int m = 0; m < 7; m++)
                        {
                            _mm_storeu_si128((__m128i *)dstPtrTemp3, px0);
                            dstPtrTemp3 += batch_srcSizeMax[batchCount].width;
                        }

                        srcPtrTemp2 += 14;
                        dstPtrTemp2 += 14;
                    }

                    if (lengthDiff > 7)
                    {
                        pixelate_base_pln_host(srcPtrChannel, batch_srcSizeMax[batchCount].width, dstPtrChannel, 7, 7, i + y1, vectorLoopCount + x1, multiplier);
                        vectorLoopCount += 7;
                    }

                    pixelate_base_pln_host(srcPtrChannel, batch_srcSizeMax[batchCount].width, dstPtrChannel, 7, lengthDiffNew, i + y1, vectorLoopCount + x1, lastColMultiplier);

                    srcPtrTemp += increment;
                    dstPtrTemp += increment;
                }

                srcPtrTemp2 = srcPtrTemp;
                dstPtrTemp2 = dstPtrTemp;

                srcPtrTemp3 = srcPtrTemp + roiEndLoc;
                dstPtrTemp3 = dstPtrTemp + roiEndLoc;

                for (int i2 = 0; i2 < heightDiff; i2++)
                {
                    memcpy(dstPtrTemp2, srcPtrTemp2, x1 * sizeof(T));
                    memcpy(dstPtrTemp3, srcPtrTemp3, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp2 += batch_srcSizeMax[batchCount].width;
                    dstPtrTemp2 += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                    dstPtrTemp3 += batch_srcSizeMax[batchCount].width;
                }

                int j = 0;
                for (; j < roiPoints[batchCount].roiWidth - lengthDiffNew; j += 7)
                {
                    pixelate_base_pln_host(srcPtrChannel, batch_srcSizeMax[batchCount].width, dstPtrChannel, heightDiff, 7, i + y1, j + x1, lastRowMultiplier);
                }

                pixelate_base_pln_host(srcPtrChannel, batch_srcSizeMax[batchCount].width, dstPtrChannel, heightDiff, lengthDiffNew, i + y1, j + x1, lastMultiplier);

                srcPtrTemp += lastIncrement;
                dstPtrTemp += lastIncrement;

                for (int i = 0; i < remainingHeight; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
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
            Rpp32u roiEndLoc = channel * (x1 + roiPoints[batchCount].roiWidth);
            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u alignedHeight = (roiPoints[batchCount].roiHeight / 7) * 7;
            Rpp32u heightDiff = roiPoints[batchCount].roiHeight - alignedHeight;

            Rpp32f multiplier = 1.0 / 49;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32u increment = elementsInRowMax * 7;
            Rpp32u lastIncrement = elementsInRowMax * heightDiff;
            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 21) * 21;
            if (alignedLength + 1 > bufferLength)
                alignedLength -= 21;
            Rpp32u lengthDiff = bufferLength - alignedLength;
            Rpp32u lengthDiffNew = lengthDiff;
            if (lengthDiff > 7)
                lengthDiffNew -= 7;

            Rpp32f lastColMultiplier = 1.0 / (lengthDiffNew * 7);
            Rpp32f lastRowMultiplier = 1.0 / (heightDiff * 7);
            Rpp32f lastMultiplier = 1.0 / (heightDiff * lengthDiffNew);

            Rpp16u vSums[64] = {0};
            Rpp16u hSums[9] = {0};

            __m128i const zero = _mm_setzero_si128();
            __m128i px0, px1, px2, px3, px4, px5, px6, px7, px8;
            __m128i pSum1, pSum2, pSum3, pSum4, pSum5, pSum6, pSum7, pSum8;

            T *srcPtrTemp, *dstPtrTemp, *srcPtrTemp2, *dstPtrTemp2, *srcPtrTemp3, *dstPtrTemp3;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for (int i = 0; i < y1; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }

            int i = 0;
            for (; i < alignedHeight; i += 7)
            {
                srcPtrTemp2 = srcPtrTemp;
                dstPtrTemp2 = dstPtrTemp;

                srcPtrTemp3 = srcPtrTemp + roiEndLoc;
                dstPtrTemp3 = dstPtrTemp + roiEndLoc;

                for (int i2 = 0; i2 < 7; i2++)
                {
                    memcpy(dstPtrTemp2, srcPtrTemp2, elementsBeforeROI * sizeof(T));
                    memcpy(dstPtrTemp3, srcPtrTemp3, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp2 += elementsInRowMax;
                    dstPtrTemp2 += elementsInRowMax;
                    srcPtrTemp3 += elementsInRowMax;
                    dstPtrTemp3 += elementsInRowMax;
                }

                srcPtrTemp2 = srcPtrTemp + elementsBeforeROI;
                dstPtrTemp2 = dstPtrTemp + elementsBeforeROI;

                Rpp32u vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=21)
                {
                    srcPtrTemp3 = srcPtrTemp2;
                    dstPtrTemp3 = dstPtrTemp2;

                    pSum1 = _mm_set1_epi16(0);
                    pSum2 = _mm_set1_epi16(0);
                    pSum3 = _mm_set1_epi16(0);
                    pSum4 = _mm_set1_epi16(0);
                    pSum5 = _mm_set1_epi16(0);
                    pSum6 = _mm_set1_epi16(0);
                    pSum7 = _mm_set1_epi16(0);
                    pSum8 = _mm_set1_epi16(0);

                    for (int m = 0; m < 7; m++)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp3);

                        px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                        px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 16));

                        px3 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        px4 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                        px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 32));

                        px5 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        px6 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                        px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 48));

                        px7 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        px8 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                        pSum1 = _mm_add_epi16(px1, pSum1);
                        pSum2 = _mm_add_epi16(px2, pSum2);
                        pSum3 = _mm_add_epi16(px3, pSum3);
                        pSum4 = _mm_add_epi16(px4, pSum4);
                        pSum5 = _mm_add_epi16(px5, pSum5);
                        pSum6 = _mm_add_epi16(px6, pSum6);
                        pSum7 = _mm_add_epi16(px7, pSum7);
                        pSum8 = _mm_add_epi16(px8, pSum8);

                        srcPtrTemp3 += elementsInRowMax;
                    }

                    _mm_storeu_si128((__m128i *)vSums, pSum1);
                    _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);
                    _mm_storeu_si128((__m128i *)(vSums + 16), pSum3);
                    _mm_storeu_si128((__m128i *)(vSums + 24), pSum4);
                    _mm_storeu_si128((__m128i *)(vSums + 32), pSum5);
                    _mm_storeu_si128((__m128i *)(vSums + 40), pSum6);
                    _mm_storeu_si128((__m128i *)(vSums + 48), pSum7);
                    _mm_storeu_si128((__m128i *)(vSums + 56), pSum8);

                    hSums[0] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[0] + vSums[3] + vSums[6] + vSums[9] + vSums[12] + vSums[15] + vSums[18]) * multiplier);
                    hSums[1] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[1] + vSums[4] + vSums[7] + vSums[10] + vSums[13] + vSums[16] + vSums[19]) * multiplier);
                    hSums[2] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[2] + vSums[5] + vSums[8] + vSums[11] + vSums[14] + vSums[17] + vSums[20]) * multiplier);

                    hSums[3] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[21] + vSums[24] + vSums[27] + vSums[30] + vSums[33] + vSums[36] + vSums[39]) * multiplier);
                    hSums[4] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[22] + vSums[25] + vSums[28] + vSums[31] + vSums[34] + vSums[37] + vSums[40]) * multiplier);
                    hSums[5] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[23] + vSums[26] + vSums[29] + vSums[32] + vSums[35] + vSums[38] + vSums[41]) * multiplier);

                    hSums[6] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[42] + vSums[45] + vSums[48] + vSums[51] + vSums[54] + vSums[57] + vSums[60]) * multiplier);
                    hSums[7] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[43] + vSums[46] + vSums[49] + vSums[52] + vSums[55] + vSums[58] + vSums[61]) * multiplier);
                    hSums[8] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[44] + vSums[47] + vSums[50] + vSums[53] + vSums[56] + vSums[59] + vSums[62]) * multiplier);

                    px1 = _mm_setr_epi16(hSums[0], hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[0], hSums[1]);
                    px2 = _mm_setr_epi16(hSums[2], hSums[0], hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[0]);
                    px3 = _mm_setr_epi16(hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[3], hSums[4], hSums[5]);
                    px4 = _mm_setr_epi16(hSums[3], hSums[4], hSums[5], hSums[3], hSums[4], hSums[5], hSums[3], hSums[4]);
                    px5 = _mm_setr_epi16(hSums[5], hSums[3], hSums[4], hSums[5], hSums[3], hSums[4], hSums[5], hSums[3]);
                    px6 = _mm_setr_epi16(hSums[4], hSums[5], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8]);
                    px7 = _mm_setr_epi16(hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7]);
                    px8 = _mm_setr_epi16(hSums[8], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8], hSums[6]);

                    px1 = _mm_packus_epi16(px1, px2);
                    px2 = _mm_packus_epi16(px3, px4);
                    px3 = _mm_packus_epi16(px5, px6);
                    px4 = _mm_packus_epi16(px7, px8);

                    for (int m = 0; m < 7; m++)
                    {
                        _mm_storeu_si128((__m128i *)dstPtrTemp3, px1);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 16), px2);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 32), px3);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 48), px4);

                        dstPtrTemp3 += elementsInRowMax;
                    }

                    srcPtrTemp2 += 63;
                    dstPtrTemp2 += 63;
                }

                for (int lengthDiffTemp = lengthDiff; lengthDiffTemp > 7; lengthDiffTemp -= 7)
                {
                    pixelate_base_pkd_host(srcPtrImage, elementsInRowMax, dstPtrImage, 7, 7, i + y1, vectorLoopCount + x1, multiplier);
                    vectorLoopCount += 7;
                }

                pixelate_base_pkd_host(srcPtrImage, elementsInRowMax, dstPtrImage, 7, lengthDiffNew, i + y1, vectorLoopCount + x1, lastColMultiplier);

                srcPtrTemp += increment;
                dstPtrTemp += increment;
            }

            srcPtrTemp2 = srcPtrTemp;
            dstPtrTemp2 = dstPtrTemp;

            srcPtrTemp3 = srcPtrTemp + roiEndLoc;
            dstPtrTemp3 = dstPtrTemp + roiEndLoc;

            for (int i2 = 0; i2 < heightDiff; i2++)
            {
                memcpy(dstPtrTemp2, srcPtrTemp2, elementsBeforeROI * sizeof(T));
                memcpy(dstPtrTemp3, srcPtrTemp3, remainingElementsAfterROI * sizeof(T));
                srcPtrTemp2 += elementsInRowMax;
                dstPtrTemp2 += elementsInRowMax;
                srcPtrTemp3 += elementsInRowMax;
                dstPtrTemp3 += elementsInRowMax;
            }

            int j = 0;
            for (; j < roiPoints[batchCount].roiWidth - lengthDiffNew; j += 7)
            {
                pixelate_base_pkd_host(srcPtrImage, elementsInRowMax, dstPtrImage, heightDiff, 7, i + y1, j + x1, lastRowMultiplier);
            }

            pixelate_base_pkd_host(srcPtrImage, elementsInRowMax, dstPtrImage, heightDiff, lengthDiffNew, i + y1, j + x1, lastMultiplier);

            srcPtrTemp += lastIncrement;
            dstPtrTemp += lastIncrement;

            for (int i = 0; i < remainingHeight; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus pixelate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u alignedHeight = (srcSize.height / 7) * 7;
    Rpp32u heightDiff = srcSize.height - alignedHeight;

    Rpp32f multiplier = 1.0 / 49;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u increment = srcSize.width * 7;
        Rpp32u bufferLength = srcSize.width;
        Rpp32u alignedLength = (bufferLength / 14) * 14;
        if (alignedLength + 2 > bufferLength)
            alignedLength -= 14;
        Rpp32u lengthDiff = bufferLength - alignedLength;
        Rpp32u lengthDiffNew = lengthDiff;
        if (lengthDiff > 7)
            lengthDiffNew -= 7;

        Rpp32f lastColMultiplier = 1.0 / (lengthDiffNew * 7);
        Rpp32f lastRowMultiplier = 1.0 / (heightDiff * 7);
        Rpp32f lastMultiplier = 1.0 / (heightDiff * lengthDiffNew);

        Rpp16u vSums[16] = {0};
        Rpp16u hSums[2] = {0};

        __m128i const zero = _mm_setzero_si128();
        __m128i px0, px1, px2;
        __m128i pSum1, pSum2;

        for (int c = 0; c < channel; c++)
        {
            T *srcPtrTemp = srcPtr + (c * imageDim);
            T *dstPtrTemp = dstPtr + (c * imageDim);

            T *srcPtrTemp2, *dstPtrTemp2, *srcPtrTemp3, *dstPtrTemp3, *srcPtrTemp4, *dstPtrTemp4;
            srcPtrTemp2 = srcPtrTemp;
            dstPtrTemp2 = dstPtrTemp;

            int i = 0;
            for (; i < alignedHeight; i += 7)
            {
                srcPtrTemp3 = srcPtrTemp2;
                dstPtrTemp3 = dstPtrTemp2;

                Rpp32u vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=14)
                {
                    srcPtrTemp4 = srcPtrTemp3;
                    dstPtrTemp4 = dstPtrTemp3;

                    pSum1 = _mm_set1_epi16(0);
                    pSum2 = _mm_set1_epi16(0);

                    for (int m = 0; m < 7; m++)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp4);

                        px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                        pSum1 = _mm_add_epi16(px1, pSum1);
                        pSum2 = _mm_add_epi16(px2, pSum2);

                        srcPtrTemp4 += srcSize.width;
                    }

                    _mm_storeu_si128((__m128i *)vSums, pSum1);
                    _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);

                    hSums[0] = vSums[0] + vSums[1] + vSums[2] + vSums[3] + vSums[4] + vSums[5] + vSums[6];
                    hSums[1] = vSums[7] + vSums[8] + vSums[9] + vSums[10] + vSums[11] + vSums[12] + vSums[13];

                    px1 = _mm_set1_epi16((Rpp16s) RPPPIXELCHECK((Rpp32f) hSums[0] * multiplier));
                    px2 = _mm_set1_epi16((Rpp16s) RPPPIXELCHECK((Rpp32f) hSums[1] * multiplier));

                    px0 = _mm_packus_epi16(px1, px2);

                    for (int m = 0; m < 7; m++)
                    {
                        _mm_storeu_si128((__m128i *)dstPtrTemp4, px0);
                        dstPtrTemp4 += srcSize.width;
                    }

                    srcPtrTemp3 += 14;
                    dstPtrTemp3 += 14;
                }

                if (lengthDiff > 7)
                {
                    pixelate_base_pln_host(srcPtrTemp, srcSize.width, dstPtrTemp, 7, 7, i, vectorLoopCount, multiplier);
                    vectorLoopCount += 7;
                }

                pixelate_base_pln_host(srcPtrTemp, srcSize.width, dstPtrTemp, 7, lengthDiffNew, i, vectorLoopCount, lastColMultiplier);

                srcPtrTemp2 += increment;
                dstPtrTemp2 += increment;
            }

            int j = 0;
            for (; j < srcSize.width - lengthDiffNew; j += 7)
            {
                pixelate_base_pln_host(srcPtrTemp, srcSize.width, dstPtrTemp, heightDiff, 7, i, j, lastRowMultiplier);
            }

            pixelate_base_pln_host(srcPtrTemp, srcSize.width, dstPtrTemp, heightDiff, lengthDiffNew, i, j, lastMultiplier);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u increment = elementsInRow * 7;

        Rpp32u bufferLength = srcSize.width;
        Rpp32u alignedLength = (bufferLength / 21) * 21;
        if (alignedLength + 1 > bufferLength)
            alignedLength -= 21;
        Rpp32u lengthDiff = bufferLength - alignedLength;
        Rpp32u lengthDiffNew = lengthDiff;
        if (lengthDiff > 7)
            lengthDiffNew -= 7;

        Rpp32f lastColMultiplier = 1.0 / (lengthDiffNew * 7);
        Rpp32f lastRowMultiplier = 1.0 / (heightDiff * 7);
        Rpp32f lastMultiplier = 1.0 / (heightDiff * lengthDiffNew);

        Rpp16u vSums[64] = {0};
        Rpp16u hSums[9] = {0};

        __m128i const zero = _mm_setzero_si128();
        __m128i px0, px1, px2, px3, px4, px5, px6, px7, px8;
        __m128i pSum1, pSum2, pSum3, pSum4, pSum5, pSum6, pSum7, pSum8;

        T *srcPtrTemp2, *dstPtrTemp2, *srcPtrTemp3, *dstPtrTemp3;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        int i = 0;
        for (; i < alignedHeight; i += 7)
        {
            srcPtrTemp2 = srcPtrTemp;
            dstPtrTemp2 = dstPtrTemp;

            Rpp32u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=21)
            {
                srcPtrTemp3 = srcPtrTemp2;
                dstPtrTemp3 = dstPtrTemp2;

                pSum1 = _mm_set1_epi16(0);
                pSum2 = _mm_set1_epi16(0);
                pSum3 = _mm_set1_epi16(0);
                pSum4 = _mm_set1_epi16(0);
                pSum5 = _mm_set1_epi16(0);
                pSum6 = _mm_set1_epi16(0);
                pSum7 = _mm_set1_epi16(0);
                pSum8 = _mm_set1_epi16(0);

                for (int m = 0; m < 7; m++)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp3);

                    px1 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    px2 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                    px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 16));

                    px3 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    px4 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                    px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 32));

                    px5 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    px6 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                    px0 =  _mm_loadu_si128((__m128i *)(srcPtrTemp3 + 48));

                    px7 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    px8 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15

                    pSum1 = _mm_add_epi16(px1, pSum1);
                    pSum2 = _mm_add_epi16(px2, pSum2);
                    pSum3 = _mm_add_epi16(px3, pSum3);
                    pSum4 = _mm_add_epi16(px4, pSum4);
                    pSum5 = _mm_add_epi16(px5, pSum5);
                    pSum6 = _mm_add_epi16(px6, pSum6);
                    pSum7 = _mm_add_epi16(px7, pSum7);
                    pSum8 = _mm_add_epi16(px8, pSum8);

                    srcPtrTemp3 += elementsInRow;
                }

                _mm_storeu_si128((__m128i *)vSums, pSum1);
                _mm_storeu_si128((__m128i *)(vSums + 8), pSum2);
                _mm_storeu_si128((__m128i *)(vSums + 16), pSum3);
                _mm_storeu_si128((__m128i *)(vSums + 24), pSum4);
                _mm_storeu_si128((__m128i *)(vSums + 32), pSum5);
                _mm_storeu_si128((__m128i *)(vSums + 40), pSum6);
                _mm_storeu_si128((__m128i *)(vSums + 48), pSum7);
                _mm_storeu_si128((__m128i *)(vSums + 56), pSum8);

                hSums[0] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[0] + vSums[3] + vSums[6] + vSums[9] + vSums[12] + vSums[15] + vSums[18]) * multiplier);
                hSums[1] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[1] + vSums[4] + vSums[7] + vSums[10] + vSums[13] + vSums[16] + vSums[19]) * multiplier);
                hSums[2] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[2] + vSums[5] + vSums[8] + vSums[11] + vSums[14] + vSums[17] + vSums[20]) * multiplier);

                hSums[3] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[21] + vSums[24] + vSums[27] + vSums[30] + vSums[33] + vSums[36] + vSums[39]) * multiplier);
                hSums[4] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[22] + vSums[25] + vSums[28] + vSums[31] + vSums[34] + vSums[37] + vSums[40]) * multiplier);
                hSums[5] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[23] + vSums[26] + vSums[29] + vSums[32] + vSums[35] + vSums[38] + vSums[41]) * multiplier);

                hSums[6] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[42] + vSums[45] + vSums[48] + vSums[51] + vSums[54] + vSums[57] + vSums[60]) * multiplier);
                hSums[7] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[43] + vSums[46] + vSums[49] + vSums[52] + vSums[55] + vSums[58] + vSums[61]) * multiplier);
                hSums[8] = (Rpp16s) RPPPIXELCHECK((Rpp32f) (vSums[44] + vSums[47] + vSums[50] + vSums[53] + vSums[56] + vSums[59] + vSums[62]) * multiplier);

                px1 = _mm_setr_epi16(hSums[0], hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[0], hSums[1]);
                px2 = _mm_setr_epi16(hSums[2], hSums[0], hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[0]);
                px3 = _mm_setr_epi16(hSums[1], hSums[2], hSums[0], hSums[1], hSums[2], hSums[3], hSums[4], hSums[5]);
                px4 = _mm_setr_epi16(hSums[3], hSums[4], hSums[5], hSums[3], hSums[4], hSums[5], hSums[3], hSums[4]);
                px5 = _mm_setr_epi16(hSums[5], hSums[3], hSums[4], hSums[5], hSums[3], hSums[4], hSums[5], hSums[3]);
                px6 = _mm_setr_epi16(hSums[4], hSums[5], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8]);
                px7 = _mm_setr_epi16(hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7]);
                px8 = _mm_setr_epi16(hSums[8], hSums[6], hSums[7], hSums[8], hSums[6], hSums[7], hSums[8], hSums[6]);

                px1 = _mm_packus_epi16(px1, px2);
                px2 = _mm_packus_epi16(px3, px4);
                px3 = _mm_packus_epi16(px5, px6);
                px4 = _mm_packus_epi16(px7, px8);

                for (int m = 0; m < 7; m++)
                {
                    _mm_storeu_si128((__m128i *)dstPtrTemp3, px1);
                    _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 16), px2);
                    _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 32), px3);
                    _mm_storeu_si128((__m128i *)(dstPtrTemp3 + 48), px4);

                    dstPtrTemp3 += elementsInRow;
                }

                srcPtrTemp2 += 63;
                dstPtrTemp2 += 63;
            }

            for (int lengthDiffTemp = lengthDiff; lengthDiffTemp > 7; lengthDiffTemp -= 7)
            {
                pixelate_base_pkd_host(srcPtr, elementsInRow, dstPtr, 7, 7, i, vectorLoopCount, multiplier);
                vectorLoopCount += 7;
            }

            pixelate_base_pkd_host(srcPtr, elementsInRow, dstPtr, 7, lengthDiffNew, i, vectorLoopCount, lastColMultiplier);


            srcPtrTemp += increment;
            dstPtrTemp += increment;
        }

        int j = 0;
        for (; j < srcSize.width - lengthDiffNew; j += 7)
        {
            pixelate_base_pkd_host(srcPtr, elementsInRow, dstPtr, heightDiff, 7, i, j, lastRowMultiplier);
        }

        pixelate_base_pkd_host(srcPtr, elementsInRow, dstPtr, heightDiff, lengthDiffNew, i, j, lastMultiplier);
    }

    return RPP_SUCCESS;
}

/**************** Fog ***************/

template <typename T>
RppStatus fog_host(  T* temp,RppiSize srcSize,T* srcPtr,
                    Rpp32f fogValue,
                    RppiChnFormat chnFormat,   unsigned int channel)
{
    Rpp8u *srcPtr1;
    srcPtr1 = srcPtr;
    for(int i = 0;i < srcSize.height * srcSize.width * channel;i++)
    {
        *srcPtr1 = *temp;
        srcPtr1++;
        temp++;
    }

    if(fogValue != 0)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp8u *srcPtr1, *srcPtr2;
            if(channel > 1)
            {
                srcPtr1 = srcPtr + (srcSize.width * srcSize.height);
                srcPtr2 = srcPtr + (srcSize.width * srcSize.height * 2);
            }
            for (int i = 0; i < (srcSize.width * srcSize.height); i++)
            {
                Rpp32f check= *srcPtr;
                if(channel > 1)
                    check = (check + *srcPtr1 + *srcPtr2) / 3;
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                srcPtr++;
                if(channel > 1)
                {
                    *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                    *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                    srcPtr1++;
                    srcPtr2++;
                }
            }
        }
        else
        {
            Rpp8u *srcPtr1, *srcPtr2;
            srcPtr1 = srcPtr + 1;
            srcPtr2 = srcPtr + 2;
            for (int i = 0; i < (srcSize.width * srcSize.height * channel); i += 3)
            {
                Rpp32f check = (*srcPtr + *srcPtr1 + *srcPtr2) / 3;
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                srcPtr += 3;
                srcPtr1 += 3;
                srcPtr2 += 3;
            }
        }
    }
    return RPP_SUCCESS;

}

template <typename T>
RppStatus fog_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32f *batch_fogValue,
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

            Rpp32f fogValue = batch_fogValue[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    if (fogValue >= 0)
                    {
                        if (channel == 3)
                        {
                            Rpp8u dstPtrTemp1, dstPtrTemp2, dstPtrTemp3;
                            dstPtrTemp1 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            dstPtrTemp2 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            dstPtrTemp3 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            Rpp32f check = (dstPtrTemp3 + dstPtrTemp1 + dstPtrTemp2) / 3;
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = fogGenerator(dstPtrTemp1, fogValue, 1, check);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = fogGenerator(dstPtrTemp2, fogValue, 2, check);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = fogGenerator(dstPtrTemp3, fogValue, 3, check);
                        }
                        if(channel == 1)
                        {
                            Rpp32f check = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = fogGenerator(check, fogValue, 1, check);
                        }
                    }
                    else
                    {
                        if (channel == 3)
                        {
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                        }
                        if(channel == 1)
                        {
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
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

            Rpp32f fogValue = batch_fogValue[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            T *srcPtrTemp1, *dstPtrTemp1;
            srcPtrTemp1 = srcPtrImage;
            dstPtrTemp1 = dstPtrImage;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    if(fogValue <= 0)
                    {
                        for(int i = 0;i < channel;i++)
                        {
                            *dstPtrTemp = *srcPtrTemp;
                            dstPtrTemp++;
                            srcPtrTemp++;
                        }
                    }
                    if(fogValue != 0)
                    {
                        Rpp8u dstPtrTemp1, dstPtrTemp2, dstPtrTemp3;
                        dstPtrTemp1 = *srcPtrTemp++;
                        dstPtrTemp2 = *srcPtrTemp++;
                        dstPtrTemp3 = *srcPtrTemp++;
                        Rpp32f check = (dstPtrTemp3 + dstPtrTemp1 + dstPtrTemp2) / 3;
                        *dstPtrTemp = fogGenerator(dstPtrTemp1, fogValue, 1, check);
                        *(dstPtrTemp+1) = fogGenerator(dstPtrTemp2, fogValue, 2, check);
                        *(dstPtrTemp+2) = fogGenerator(dstPtrTemp3, fogValue, 3, check);
                        dstPtrTemp += channel;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Noise ***************/

template <typename T>
RppStatus noise_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        Rpp32f noiseProbability,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    T *dstPtrTemp;

    srand(time(0));
    int seedVal[5000];

    for (int i = 0; i < 5000; i++)
    {
        seedVal[i] = rand() % (65536);
    }

    memcpy(dstPtr, srcPtr, channel * srcSize.width * srcSize.height * sizeof(T));

    Rpp32u imageDim = srcSize.width * srcSize.height * channel;

    if(noiseProbability != 0)
    {
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            int count = 0;
            Rpp32u increment = pixelDistance * channel;
            Rpp32u limit = imageDim - (2 * increment);

            for(int i = 0 ; i < limit ; i += increment)
            {
                wyhash16_x = seedVal[count];
                Rpp32u initialPixel = rand_range16((uint16_t) pixelDistance);
                dstPtrTemp = dstPtr + (initialPixel * channel);
                Rpp8u newPixel = rand_range16(2) ? 0 : 255;
                for(int j = 0 ; j < channel ; j++)
                {
                    *dstPtrTemp = newPixel;
                    dstPtrTemp++;
                }
                dstPtr += increment;
                (count == 5000) ? count = 0 : count++;
            }
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            int count = 0;
            Rpp32u increment = pixelDistance;
            Rpp32u limit = imageDim - (2 * increment);

            if(channel == 3)
            {
                Rpp8u *dstPtrTemp1,*dstPtrTemp2;
                dstPtrTemp1 = dstPtr + (srcSize.height * srcSize.width);
                dstPtrTemp2 = dstPtr + (2 * srcSize.height * srcSize.width);
                for(int i = 0 ; i < limit ; i += pixelDistance)
                {
                    Rpp32u initialPixel = rand() % pixelDistance;
                    dstPtr += initialPixel;
                    Rpp8u newPixel = (rand() % 2) ? 255 : 1;
                    *dstPtr = newPixel;
                    dstPtr += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp1 += initialPixel;
                    *dstPtrTemp1 = newPixel;
                    dstPtrTemp1 += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp2 += initialPixel;
                    *dstPtrTemp2 = newPixel;
                    dstPtrTemp2 += ((pixelDistance - initialPixel - 1));

                }
            }
            else
            {
                for(int i = 0 ; i < srcSize.width * srcSize.height ; i += pixelDistance)
                {
                    wyhash16_x = seedVal[count];
                    Rpp32u initialPixel = rand_range16((uint16_t) pixelDistance);//rand() % pixelDistance;
                    dstPtrTemp = dstPtr + initialPixel;
                    Rpp8u newPixel = rand_range16(2) ? 255 : 1;
                    *dstPtrTemp = newPixel;
                    dstPtr += increment;
                    (count == 5000) ? count = 0 : count++;
                }
            }

        }
    }
    return RPP_SUCCESS;
}

template <typename T>
RppStatus noise_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                           Rpp32f *batch_noiseProbability,
                           RppiROI *roiPoints, Rpp32u nbatchSize,
                           RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrBufferROI, *dstPtrBufferROI;
        srcPtrBufferROI = (T*) calloc(channel * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * nbatchSize, sizeof(T));
        dstPtrBufferROI = (T*) calloc(channel * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * nbatchSize, sizeof(T));

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

            RppiSize roiSize;
            roiSize.height = roiPoints[batchCount].roiHeight;
            roiSize.width = roiPoints[batchCount].roiWidth;

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));
            Rpp32u remainingElementsAfterROIMax = (batch_srcSizeMax[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));
            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1);

            Rpp32f noiseProbability = batch_noiseProbability[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u imageDimROI = roiSize.width * roiSize.height;

            T *srcPtrImageROI, *dstPtrImageROI;
            srcPtrImageROI = srcPtrBufferROI + loc;
            dstPtrImageROI = dstPtrBufferROI + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                T *srcPtrTemp, *dstPtrTemp, *srcPtrTemp2, *dstPtrTemp2;
                srcPtrTemp = srcPtrChannel;
                dstPtrTemp = dstPtrChannel;

                T *srcPtrImageROITemp, *dstPtrImageROITemp;
                srcPtrImageROITemp = srcPtrImageROI;
                dstPtrImageROITemp = dstPtrImageROI;

                for (int i = 0; i < y1; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }

                srcPtrTemp2 = srcPtrTemp + x1;

                for (int i = 0; i < roiSize.height; i++)
                {
                    memcpy(srcPtrImageROITemp, srcPtrTemp2, roiSize.width * sizeof(T));
                    srcPtrTemp2 += batch_srcSizeMax[batchCount].width;
                    srcPtrImageROITemp += roiSize.width;
                }

                noise_host(srcPtrImageROI, roiSize, dstPtrImageROI, noiseProbability, chnFormat, 1);

                for (int i = 0; i < roiSize.height; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                    srcPtrTemp += x1;
                    dstPtrTemp += x1;
                    memcpy(dstPtrTemp, dstPtrImageROITemp, roiSize.width * sizeof(T));
                    srcPtrTemp += roiSize.width;
                    dstPtrTemp += roiSize.width;
                    dstPtrImageROITemp += roiSize.width;
                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROIMax;
                    dstPtrTemp += remainingElementsAfterROIMax;
                }

                for (int i = 0; i < remainingHeight; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }
            }
        }

        free(srcPtrBufferROI);
        free(dstPtrBufferROI);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrBufferROI, *dstPtrBufferROI;
        srcPtrBufferROI = (T*) calloc(channel * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * nbatchSize, sizeof(T));
        dstPtrBufferROI = (T*) calloc(channel * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * nbatchSize, sizeof(T));

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

            RppiSize roiSize;
            roiSize.height = roiPoints[batchCount].roiHeight;
            roiSize.width = roiPoints[batchCount].roiWidth;

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));
            Rpp32u remainingElementsAfterROIMax = channel * (batch_srcSizeMax[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));
            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1);

            Rpp32f noiseProbability = batch_noiseProbability[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u imageDimROI = channel * roiSize.width * roiSize.height;

            T *srcPtrImageROI, *dstPtrImageROI;
            srcPtrImageROI = srcPtrBufferROI + loc;
            dstPtrImageROI = dstPtrBufferROI + loc;

            T *srcPtrImageROITemp, *dstPtrImageROITemp;
            srcPtrImageROITemp = srcPtrImageROI;
            dstPtrImageROITemp = dstPtrImageROI;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * roiSize.width;

            T *srcPtrTemp, *dstPtrTemp, *srcPtrTemp2, *dstPtrTemp2;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for (int i = 0; i < y1; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }

            srcPtrTemp2 = srcPtrTemp + elementsBeforeROI;

            for (int i = 0; i < roiSize.height; i++)
            {
                memcpy(srcPtrImageROITemp, srcPtrTemp2, elementsInRowROI * sizeof(T));
                srcPtrTemp2 += elementsInRowMax;
                srcPtrImageROITemp += elementsInRowROI;
            }

            noise_host(srcPtrImageROI, roiSize, dstPtrImageROI, noiseProbability, chnFormat, 3);

            for (int i = 0; i < roiSize.height; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                srcPtrTemp += elementsBeforeROI;
                dstPtrTemp += elementsBeforeROI;
                memcpy(dstPtrTemp, dstPtrImageROITemp, elementsInRowROI * sizeof(T));
                srcPtrTemp += elementsInRowROI;
                dstPtrTemp += elementsInRowROI;
                dstPtrImageROITemp += elementsInRowROI;
                memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                srcPtrTemp += remainingElementsAfterROIMax;
                dstPtrTemp += remainingElementsAfterROIMax;
            }

            for (int i = 0; i < remainingHeight; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }
        }

        free(srcPtrBufferROI);
        free(dstPtrBufferROI);
    }

    return RPP_SUCCESS;
}

/**************** Snow ***************/

template <typename T>
RppStatus snow_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f strength,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    strength = strength/100;
    int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

    Rpp32u snowDrops = (Rpp32u)(strength * srcSize.width * srcSize.height * channel );

    T *dstptrtemp;
    dstptrtemp=dstPtr;
    for(int k=0;k<srcSize.height*srcSize.width*channel;k++)
    {
        *dstptrtemp = 0;
        dstptrtemp++;
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width)] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width) + (srcSize.width * j) + m] = snow_mat[j][m] ;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j) + (channel * m)] = snow_mat[j][m];
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32u pixel = ((Rpp32u) srcPtr[i]) + (Rpp32u)dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus snow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32f *batch_strength,
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
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );


            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] = RPPPIXELCHECK(dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSizeMax[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] = RPPPIXELCHECK( dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] + snow_mat[j][m]) ;
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
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );

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
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] = RPPPIXELCHECK(dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSize[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] = RPPPIXELCHECK( dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] + snow_mat[j][m]);
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Rain ***************/

template <typename T>
RppStatus rain_host(T* srcPtr, RppiSize srcSize,T* dstPtr,
                    Rpp32f rainPercentage, Rpp32f rainWidth, Rpp32f rainHeight, Rpp32f transparency,
                    RppiChnFormat chnFormat,   unsigned int channel)
{
    rainPercentage *= 0.004;
    transparency *= 0.2;

    const Rpp32u rainDrops = (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel);
    fast_srand(time(0));
    const unsigned rand_len = srcSize.width;
    unsigned int col_rand[rand_len];
    unsigned int row_rand[rand_len];
    for(int i = 0; i<  rand_len; i++)
    {
        col_rand[i] = fastrand() % srcSize.width;
        row_rand[i] = fastrand() % srcSize.height;
    }

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            int rand_idx = i%rand_len;
            Rpp32u row = row_rand[rand_idx];
            Rpp32u column = col_rand[rand_idx];
            Rpp32f pixel;
            Rpp8u *dst = &dstPtr[(row * srcSize.width) + column];
            Rpp8u *dst1, *dst2;
            if (channel > 1){
                dst1 = dst + srcSize.width*srcSize.height;
                dst2 = dst1 + srcSize.width*srcSize.height;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                for(int m = 0;m < rainWidth;m++)
                {
                    if ( (row + rainHeight) < srcSize.height && (column + rainWidth) < srcSize.width)
                    {
                        int idx = srcSize.width*j + m;
                        dst[idx] = 196;
                        if (channel > 1) {
                            dst1[idx] = 226, dst2[idx] = 255;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            int rand_idx = i%rand_len;
            Rpp32u row = row_rand[rand_idx];
            Rpp32u column = col_rand[rand_idx];
            Rpp32f pixel;
            Rpp8u *dst = &dstPtr[(row * srcSize.width*channel) + column*channel];
            for(int j = 0;j < rainHeight; j++)
            {
                for(int m = 0;m < rainWidth; m++)
                {
                    if ((row + rainHeight) < srcSize.height && (column + rainWidth) < srcSize.width )
                    {
                        int idx = (j*srcSize.width*channel) + m*channel;
                        dst[idx] = 196;
                        if (channel > 1) {
                            dst[idx+1] = 226;
                            dst[idx+2] = 255;
                        }
                    }
                }
            }
        }
    }

    Rpp8u *src = &srcPtr[0];
    Rpp8u *dst = &dstPtr[0];
    int length = channel * srcSize.width * srcSize.height;
    int i=0;
#if __AVX2__
    int alignedLength = length & ~31;
    __m256i const trans = _mm256_set1_epi16((unsigned short)(transparency*65535));       // 1/5th
    __m256i const zero = _mm256_setzero_si256();
    for (; i < alignedLength; i+=32)
    {
        __m256i s0, s1, r0;
        s0 = _mm256_loadu_si256((__m256i *) dst);
        r0 = _mm256_loadu_si256((__m256i *) src);
        s1 = _mm256_unpacklo_epi8(s0, zero);
        s0 = _mm256_unpackhi_epi8(s0, zero);
        s1 = _mm256_mulhi_epi16(s1, trans);
        s0 = _mm256_mulhi_epi16(s0, trans);
        s1 = _mm256_add_epi16(s1, _mm256_unpacklo_epi8(r0, zero));
        s0 = _mm256_add_epi16(s0, _mm256_unpackhi_epi8(r0, zero));
        _mm256_storeu_si256((__m256i *)dst, _mm256_packus_epi16(s1, s0));
        dst += 32;
        src += 32;
    }
#else
    int alignedLength = length & ~15;
    __m128i const trans = _mm_set1_epi16((unsigned short)(transparency*65535));       // trans factor
    __m128i const zero = _mm_setzero_si128();
    for (; i < alignedLength; i+=16)
    {
        __m128i s0, s1, r0;
        s0 = _mm_loadu_si128((__m128i *) dst);
        r0 = _mm_loadu_si128((__m128i *) src);
        s1 = _mm_unpacklo_epi8(s0, zero);
        s0 = _mm_unpackhi_epi8(s0, zero);
        s1 = _mm_mulhi_epi16(s1, trans);
        s0 = _mm_mulhi_epi16(s0, trans);
        s1 = _mm_add_epi16(s1, _mm_unpacklo_epi8(r0, zero));
        s0 = _mm_add_epi16(s0, _mm_unpackhi_epi8(r0, zero));
        _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(s1, s0));
        dst += 16;
        src += 16;
    }
#endif

    return RPP_SUCCESS;
}

template <typename T>
RppStatus rain_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                          Rpp32f *batch_rainPercentage, Rpp32u *batch_rainWidth, Rpp32u *batch_rainHeight, Rpp32f *batch_transparency,
                          Rpp32u nbatchSize, RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32f rainPercentage = batch_rainPercentage[batchCount];
        Rpp32u rainWidth = batch_rainWidth[batchCount];
        Rpp32u rainHeight = batch_rainHeight[batchCount];
        Rpp32f transparency = batch_transparency[batchCount];

        T *srcPtrImage, *dstPtrImage;
        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
        srcPtrImage = srcPtr + loc;
        dstPtrImage = dstPtr + loc;

        T *srcPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImageUnpadded, chnFormat, channel);

        rain_host(srcPtrImageUnpadded, batch_srcSize[batchCount], dstPtrImageUnpadded, rainPercentage, rainWidth, rainHeight, transparency, chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, chnFormat, channel);

        free(srcPtrImageUnpadded);
        free(dstPtrImageUnpadded);
    }

    return RPP_SUCCESS;
}

/**************** Random Shadow ***************/

template <typename T>
RppStatus random_shadow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                             Rpp32u *batch_x1, Rpp32u *batch_y1, Rpp32u *batch_x2, Rpp32u *batch_y2,
                             Rpp32u *batch_numberOfShadows, Rpp32u *batch_maxSizeX, Rpp32u *batch_maxSizeY,
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

            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u numberOfShadows = batch_numberOfShadows[batchCount];
            Rpp32u maxSizeX = batch_maxSizeX[batchCount];
            Rpp32u maxSizeY = batch_maxSizeY[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            }

            srand (time(NULL));
            RppiSize srcSizeSubImage,shadowSize;
            T *srcPtrSubImage, *dstPtrSubImage;
            srcSizeSubImage.height = RPPABS(y2 - y1) + 1;
            srcSizeSubImage.width = RPPABS(x2 - x1) + 1;
            srcPtrSubImage = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);
            dstPtrSubImage = dstPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;


            for (int shadow = 0; shadow < numberOfShadows; shadow++)
            {
                shadowSize.height = rand() % maxSizeY;
                shadowSize.width = rand() % maxSizeX;
                Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
                Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
                Rpp32u remainingElementsInRow = batch_srcSizeMax[batchCount].width - shadowSize.width;
                for (int c = 0; c < channel; c++)
                {
                    dstPtrTemp = dstPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;
                    srcPtrTemp = srcPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;

                    for (int i = 0; i < shadowSize.height; i++)
                    {
                        for (int j = 0; j < shadowSize.width; j++)
                        {
                            *dstPtrTemp = *srcPtrTemp / 2;
                            dstPtrTemp++;
                            srcPtrTemp++;
                        }
                        dstPtrTemp += remainingElementsInRow;
                        srcPtrTemp += remainingElementsInRow;
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

            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u numberOfShadows = batch_numberOfShadows[batchCount];
            Rpp32u maxSizeX = batch_maxSizeX[batchCount];
            Rpp32u maxSizeY = batch_maxSizeY[batchCount];
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

            srand (time(NULL));
            RppiSize srcSizeSubImage, shadowSize;
            T *srcPtrSubImage, *dstPtrSubImage;
            srcSizeSubImage.height = RPPABS(y2 - y1) + 1;
            srcSizeSubImage.width = RPPABS(x2 - x1) + 1;
            srcPtrSubImage = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width * channel) + (x1 * channel);
            dstPtrSubImage = dstPtrImage + (y1 * batch_srcSizeMax[batchCount].width * channel) + (x1 * channel);

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;


            for (int shadow = 0; shadow < numberOfShadows; shadow++)
            {
                shadowSize.height = rand() % maxSizeY;
                shadowSize.width = rand() % maxSizeX;
                Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
                Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
                Rpp32u remainingElementsInRow = channel * (batch_srcSizeMax[batchCount].width - shadowSize.width);
                dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ));
                srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ));
                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *srcPtrTemp / 2;
                            dstPtrTemp++;
                            srcPtrTemp++;
                        }
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus random_shadow_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                             Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                             Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    srand (time(NULL));
    RppiSize srcSizeSubImage, dstSizeSubImage, shadowSize;
    T *srcPtrSubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    if (maxSizeX > srcSizeSubImage.width || maxSizeY > srcSizeSubImage.height)
    {
        return RPP_ERROR;
    }

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &dstSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    memcpy(dstPtr, srcPtr, channel * srcSize.height * srcSize.width * sizeof(T));

    for (int shadow = 0; shadow < numberOfShadows; shadow++)
    {
        shadowSize.height = rand() % maxSizeY;
        shadowSize.width = rand() % maxSizeX;
        Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
        Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp32u remainingElementsInRow = srcSize.width - shadowSize.width;
            for (int c = 0; c < channel; c++)
            {
                dstPtrTemp = dstPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;
                srcPtrTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;

                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            Rpp32u remainingElementsInRow = channel * (srcSize.width - shadowSize.width);
            for (int i = 0; i < shadowSize.height; i++)
            {
                for (int j = 0; j < shadowSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                }
                dstPtrTemp += remainingElementsInRow;
                srcPtrTemp += remainingElementsInRow;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Jitter ***************/

template <typename T>
RppStatus jitter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32u bound = (kernelSize - 1);
            Rpp32u widthLimit = roiPoints[batchCount].roiWidth - bound;
            Rpp32u heightLimit = roiPoints[batchCount].roiHeight - bound;

            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1) + bound;

            srand(time(0));
            int seedVal;

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

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel;
                dstPtrTemp = dstPtrChannel;

                for (int i = 0; i < y1; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }

                for(int i = 0 ; i < heightLimit; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                    srcPtrTemp += x1;
                    dstPtrTemp += x1;

                    seedVal = rand() % (65536);

                    for(int j = 0 ; j < widthLimit; j++)
                    {
                        wyhash16_x = seedVal;
                        Rpp16u nhx = rand_range16(kernelSize);
                        Rpp16u nhy = rand_range16(kernelSize);

                        *dstPtrTemp = *(srcPtrChannel + ((y1 + i + nhy) * batch_srcSizeMax[batchCount].width) + (x1 + j + nhx));
                        dstPtrTemp++;
                    }
                    srcPtrTemp += widthLimit;
                    memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                    dstPtrTemp += bound;
                    srcPtrTemp += bound;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }

                for (int i = 0; i < remainingHeight; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32u bound = (kernelSize - 1);
            Rpp32u widthLimit = roiPoints[batchCount].roiWidth - bound;
            Rpp32u heightLimit = roiPoints[batchCount].roiHeight - bound;

            Rpp32u remainingHeight = batch_srcSize[batchCount].height - (y2 + 1) + bound;

            srand(time(0));
            int seedVal;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u remainingElements = bound * channel;
            Rpp32u increment = channel * widthLimit;

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for (int i = 0; i < y1; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }

            for(int i = 0 ; i < heightLimit; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                srcPtrTemp += elementsBeforeROI;
                dstPtrTemp += elementsBeforeROI;

                seedVal = rand() % (65536);

                for(int j = 0 ; j < widthLimit; j++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);

                    for(int c = 0 ; c < channel ; c++)
                    {
                        *dstPtrTemp = *(srcPtrImage + ((y1 + i + nhy) * elementsInRow) + ((x1 + j + nhx) * channel) + c);
                        dstPtrTemp++;
                    }
                }
                srcPtrTemp += increment;
                memcpy(dstPtrTemp, srcPtrTemp, remainingElements * sizeof(T));
                dstPtrTemp += remainingElements;
                srcPtrTemp += remainingElements;

                memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                srcPtrTemp += remainingElementsAfterROI;
                dstPtrTemp += remainingElementsAfterROI;
            }

            for (int i = 0; i < remainingHeight; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus jitter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32u bound = (kernelSize - 1);
    Rpp32u widthLimit = srcSize.width - bound;
    Rpp32u heightLimit = srcSize.height - bound;

    srand(time(0));
    int seedVal;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u imageDim = srcSize.height * srcSize.width;

        T *srcPtrChannel, *dstPtrChannel;

        for (int c = 0; c < channel; c++)
        {
            srcPtrChannel = srcPtr + (c * imageDim);
            dstPtrChannel = dstPtr + (c * imageDim);

            srcPtrTemp = srcPtrChannel;
            dstPtrTemp = dstPtrChannel;

            for(int i = 0 ; i < heightLimit; i++)
            {
                seedVal = rand() % (65536);
                for(int j = 0 ; j < widthLimit; j++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);

                    *dstPtrTemp = *(srcPtrChannel + ((i + nhy) * srcSize.width) + (j + nhx));
                    dstPtrTemp++;
                }
                srcPtrTemp += widthLimit;
                memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                dstPtrTemp += bound;
                srcPtrTemp += bound;
            }
            memcpy(dstPtrTemp, srcPtrTemp, bound * srcSize.width * sizeof(T));
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u remainingElements = bound * channel;
        Rpp32u increment = channel * widthLimit;

        for(int i = 0 ; i < heightLimit; i++)
        {
            seedVal = rand() % (65536);
            for(int j = 0 ; j < widthLimit; j++)
            {
                wyhash16_x = seedVal;
                Rpp16u nhx = rand_range16(kernelSize);
                Rpp16u nhy = rand_range16(kernelSize);

                for(int c = 0 ; c < channel ; c++)
                {
                    *dstPtrTemp = *(srcPtr + ((i + nhy) * elementsInRow) + ((j + nhx) * channel) + c);
                    dstPtrTemp++;
                }
            }
            srcPtrTemp += increment;
            memcpy(dstPtrTemp, srcPtrTemp, remainingElements * sizeof(T));
            dstPtrTemp += remainingElements;
            srcPtrTemp += remainingElements;
        }
        memcpy(dstPtrTemp, srcPtrTemp, bound * elementsInRow * sizeof(T));
    }

    return RPP_SUCCESS;
}

#endif
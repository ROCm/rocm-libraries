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

#include "host_tensor_executors.hpp"

inline void reduce_max_32_host(__m256i *pMax, __m128i *result)
{
    __m128i px;
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px = _mm_max_epu8(_mm256_castsi256_si128(pMax[0]), _mm256_extracti128_si256(pMax[0], 1));
    px = _mm_max_epu8(_mm_unpacklo_epi8(zero, px), _mm_unpackhi_epi8(zero, px));
    px = _mm_max_epu16(_mm_unpacklo_epi16(zero, px), _mm_unpackhi_epi16(zero, px));
    px = _mm_max_epu32(_mm_unpacklo_epi32(zero, px), _mm_unpackhi_epi32(zero, px));
    result[0] = _mm_shuffle_epi8(px, mask);
}

inline void compute_max_96_host(__m256i *p1, __m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB)
{
    pMaxR[0] = _mm256_max_epu8(p1[0], pMaxR[0]); //compare and store max of 32 R values into global max
    pMaxG[0] = _mm256_max_epu8(p1[1], pMaxG[0]); //compare and store max of 32 G values into global max
    pMaxB[0] = _mm256_max_epu8(p1[2], pMaxB[0]); //compare and store max of 32 B values into global max
}

inline void reduce_max_96_host(__m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_max_epu8(_mm256_castsi256_si128(pMaxR[0]), _mm256_extracti128_si256(pMaxR[0], 1));
    px[1] = _mm_max_epu8(_mm256_castsi256_si128(pMaxG[0]), _mm256_extracti128_si256(pMaxG[0], 1));
    px[1] = _mm_max_epu8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_max_epu8(_mm256_castsi256_si128(pMaxB[0]), _mm256_extracti128_si256(pMaxB[0], 1));
    px[0] = _mm_max_epu8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_max_epu8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epu8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epu8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_max_48_host(__m128i *p1, __m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB)
{
    pMaxR[0] = _mm_max_epu8(p1[0], pMaxR[0]); //compare and store max of 16 R values into global max
    pMaxG[0] = _mm_max_epu8(p1[1], pMaxG[0]); //compare and store max of 16 G values into global max
    pMaxB[0] = _mm_max_epu8(p1[2], pMaxB[0]); //compare and store max of 16 B values into global max
}

inline void reduce_max_48_host(__m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxR[0], pMaxG[0]), _mm_unpackhi_epi8(pMaxR[0], pMaxG[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxB[0], zero), _mm_unpackhi_epi8(pMaxB[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}
inline void compute_max_float8_host(__m256 *p1, __m256 *pMax)
{
    pMax[0] = _mm256_max_ps(p1[0], pMax[0]); //compare and store max of 8 values into global min
}

inline void reduce_max_float8_host(__m256 *pMax, __m128 *result)
{
    __m128 px;
    px = _mm_max_ps(_mm256_castps256_ps128(pMax[0]), _mm256_extractf128_ps(pMax[0], 1));
    px = _mm_max_ps(_mm_unpacklo_ps(xmm_p0, px), _mm_unpackhi_ps(xmm_p0, px));
    result[0] = _mm_shuffle_ps(px, px, 39);
}

inline void compute_max_float24_host(__m256 *p1, __m256 *pMaxR, __m256 *pMaxG, __m256 *pMaxB)
{
    pMaxR[0] = _mm256_max_ps(p1[0], pMaxR[0]); //compare and store max of 8 R values into global min
    pMaxG[0] = _mm256_max_ps(p1[1], pMaxG[0]); //compare and store max of 8 G values into global min
    pMaxB[0] = _mm256_max_ps(p1[2], pMaxB[0]); //compare and store max of 8 B values into global min
}

inline void reduce_max_float24_host(__m256 *pMaxR, __m256 *pMaxG, __m256 *pMaxB, __m256 *result)
{
    __m128 px[2];
    px[0] = _mm_max_ps(_mm256_castps256_ps128(pMaxR[0]), _mm256_extractf128_ps(pMaxR[0], 1));
    px[1] = _mm_max_ps(_mm256_castps256_ps128(pMaxG[0]), _mm256_extractf128_ps(pMaxG[0], 1));
    px[0] = _mm_max_ps(_mm_unpacklo_ps(px[0], px[1]), _mm_unpackhi_ps(px[0], px[1]));
    px[0] = _mm_permute_ps(px[0], 0b11011000);
    result[0] = _mm256_castps128_ps256(px[0]);
    px[0] = _mm_max_ps(_mm256_castps256_ps128(pMaxB[0]), _mm256_extractf128_ps(pMaxB[0], 1));
    px[1] = _mm_max_ps(_mm_unpacklo_ps(px[0], xmm_p0), _mm_unpackhi_ps(px[0], xmm_p0));
    px[0] = _mm_shuffle_ps(px[1], px[1], 34);
    result[0] = _mm256_insertf128_ps(result[0], px[0], 1);
}

inline void reduce_max_i32_host(__m256i *pMax, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMax[0]), _mm256_extracti128_si256(pMax[0], 1));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(zero, px[0]), _mm_unpackhi_epi8(zero, px[0]));
    px[0] = _mm_max_epi16(_mm_unpacklo_epi16(zero, px[0]), _mm_unpackhi_epi16(zero, px[0]));
    px[0] = _mm_max_epi32(_mm_unpacklo_epi32(zero, px[0]), _mm_unpackhi_epi32(zero, px[0]));
    result[0] = _mm_shuffle_epi8(px[0], mask);
}

inline void compute_max_i96_host(__m256i *p1, __m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB)
{
    pMaxR[0] = _mm256_max_epi8(p1[0], pMaxR[0]); //compare and store max of 32 R values into global max
    pMaxG[0] = _mm256_max_epi8(p1[1], pMaxG[0]); //compare and store max of 32 G values into global max
    pMaxB[0] = _mm256_max_epi8(p1[2], pMaxB[0]); //compare and store max of 32 B values into global max
}

inline void reduce_max_i96_host(__m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMaxR[0]), _mm256_extracti128_si256(pMaxR[0], 1));
    px[1] = _mm_max_epi8(_mm256_castsi256_si128(pMaxG[0]), _mm256_extracti128_si256(pMaxG[0], 1));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMaxB[0]), _mm256_extracti128_si256(pMaxB[0], 1));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_max_i48_host(__m128i *p1, __m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB)
{
    pMaxR[0] = _mm_max_epi8(p1[0], pMaxR[0]); //compare and store max of 16 R values into global max
    pMaxG[0] = _mm_max_epi8(p1[1], pMaxG[0]); //compare and store max of 16 G values into global max
    pMaxB[0] = _mm_max_epi8(p1[2], pMaxB[0]); //compare and store max of 16 B values into global max
}

inline void reduce_max_i48_host(__m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxR[0], pMaxG[0]), _mm_unpackhi_epi8(pMaxR[0], pMaxG[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxB[0], zero), _mm_unpackhi_epi8(pMaxB[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

RppStatus tensor_max_u8_u8_host(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *maxArr,
                                Rpp32u maxArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        // Tensor max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8u max = 0;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256i pMax = _mm256_setzero_si256();
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((__m256i *)srcPtrTemp);
                        pMax = _mm256_max_epu8(p1, pMax); //compare and store max of 32 values into global max

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        max = std::max(*srcPtrTemp++, max);
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_32_host(&pMax, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                max = std::max(resultAvx[0], max);
#endif
            maxArr[batchCount] = max;
        }
        // Tensor max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp8u maxC = 0, maxR = 0, maxG = 0, maxB = 0;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
                __m256i pMaxR = _mm256_setzero_si256();
                __m256i pMaxG = pMaxR;
                __m256i pMaxB = pMaxR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256i p[3];
                        rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                        compute_max_96_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        maxR = std::max(*srcPtrTempR++, maxR);
                        maxG = std::max(*srcPtrTempG++, maxG);
                        maxB = std::max(*srcPtrTempB++, maxB);
                    }
                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_96_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                maxR = std::max(resultAvx[0], maxR);
                maxG = std::max(resultAvx[1], maxG);
                maxB = std::max(resultAvx[2], maxB);
#endif
            }
            maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
            maxArr[maxArrIndex + 1] = maxG;
            maxArr[maxArrIndex + 2] = maxB;
            maxArr[maxArrIndex + 3] = maxC;
        }

        // Tensor max 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8u maxC = 0, maxR = 0, maxG = 0, maxB = 0;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;

                __m128i pMaxR = _mm_setzero_si128();
                __m128i pMaxG = pMaxR;
                __m128i pMaxB = pMaxR;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i p[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, p);
                        compute_max_48_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        maxR = std::max(srcPtrTemp[0], maxR);
                        maxG = std::max(srcPtrTemp[1], maxG);
                        maxB = std::max(srcPtrTemp[2], maxB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_48_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                maxR = std::max(resultAvx[0], maxR);
                maxG = std::max(resultAvx[1], maxG);
                maxB = std::max(resultAvx[2], maxB);
#endif
            }
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
			maxArr[maxArrIndex + 1] = maxG;
			maxArr[maxArrIndex + 2] = maxB;
			maxArr[maxArrIndex + 3] = maxC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_max_f32_f32_host(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *maxArr,
                                  Rpp32u maxArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Tensor max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp32f max = 0.0;
            Rpp32f resultAvx[4];

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 pMax = _mm256_setzero_ps();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p1);
                    compute_max_float8_host(&p1, &pMax);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    max = std::max(*srcPtrTemp++, max);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m128 result;
            reduce_max_float8_host(&pMax, &result);
            rpp_simd_store(rpp_store4_f32_to_f32, resultAvx, &result);
            max = std::max(std::max(resultAvx[0], resultAvx[1]), max);
#endif
            maxArr[batchCount] = max;
        }

        // Tensor max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32f maxC = 0.0, maxR = 0.0, maxG = 0.0, maxB = 0.0;
            Rpp32f resultAvx[8];

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256 pMaxR = _mm256_setzero_ps();
            __m256 pMaxG = pMaxR;
            __m256 pMaxB = pMaxR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_max_float24_host(p, &pMaxR, &pMaxG, &pMaxB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    maxR = std::max(*srcPtrTempR++, maxR);
                    maxG = std::max(*srcPtrTempG++, maxG);
                    maxB = std::max(*srcPtrTempB++, maxB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m256 result;
            reduce_max_float24_host(&pMaxR, &pMaxG, &pMaxB, &result);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

            maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
            maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
            maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);
#endif
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
			maxArr[maxArrIndex + 1] = maxG;
			maxArr[maxArrIndex + 2] = maxB;
			maxArr[maxArrIndex + 3] = maxC;
        }

        // Tensor max 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32u vectorIncrement = 24;
            Rpp32f maxC = 0.0, maxR = 0.0, maxG = 0.0, maxB = 0.0;
            Rpp32f resultAvx[8];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX2__
                __m256 pMaxR = _mm256_setzero_ps();
                __m256 pMaxG = pMaxR;
                __m256 pMaxB = pMaxR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                        compute_max_float24_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        maxR = std::max(srcPtrTemp[0], maxR);
                        maxG = std::max(srcPtrTemp[1], maxG);
                        maxB = std::max(srcPtrTemp[2], maxB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m256 result;
                reduce_max_float24_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

                maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
                maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
                maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);
#endif
            }
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
			maxArr[maxArrIndex + 1] = maxG;
			maxArr[maxArrIndex + 2] = maxB;
			maxArr[maxArrIndex + 3] = maxC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_max_f16_f16_host(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *maxArr,
                                  Rpp32u maxArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Tensor max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp32f max = 0.0;
            Rpp32f resultAvx[4];

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 pMax = _mm256_setzero_ps();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                    {
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
                    }
                    __m256 p1;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, &p1);
                    compute_max_float8_host(&p1, &pMax);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    max = std::max((Rpp32f)*srcPtrTemp++, max);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m128 result;
            reduce_max_float8_host(&pMax, &result);
            rpp_simd_store(rpp_store4_f32_to_f32, resultAvx, &result);
            max = std::max(std::max(resultAvx[0], resultAvx[1]), max);
#endif
            maxArr[batchCount] = (Rpp16f)max;
        }

        // Tensor max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32f maxC = 0.0, maxR = 0.0, maxG = 0.0, maxB = 0.0;
            Rpp32f resultAvx[8];

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256 pMaxR = _mm256_setzero_ps();
            __m256 pMaxG = pMaxR;
            __m256 pMaxB = pMaxR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);
                    compute_max_float24_host(p, &pMaxR, &pMaxG, &pMaxB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    maxR = std::max((Rpp32f)*srcPtrTempR++, maxR);
                    maxG = std::max((Rpp32f)*srcPtrTempG++, maxG);
                    maxB = std::max((Rpp32f)*srcPtrTempB++, maxB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m256 result;
            reduce_max_float24_host(&pMaxR, &pMaxG, &pMaxB, &result);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

            maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
            maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
            maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);

#endif
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = (Rpp16f)maxR;
			maxArr[maxArrIndex + 1] = (Rpp16f)maxG;
			maxArr[maxArrIndex + 2] = (Rpp16f)maxB;
			maxArr[maxArrIndex + 3] = (Rpp16f)maxC;
        }

        // Tensor max 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32u vectorIncrement = 24;
            Rpp32f maxC = 0.0, maxR = 0.0, maxG = 0.0, maxB = 0.0;
            Rpp32f resultAvx[8];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX2__
                __m256 pMaxR = _mm256_setzero_ps();
                __m256 pMaxG = pMaxR;
                __m256 pMaxB = pMaxR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[24];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        {
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
                        }
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);
                        compute_max_float24_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        maxR = std::max((Rpp32f)srcPtrTemp[0], maxR);
                        maxG = std::max((Rpp32f)srcPtrTemp[1], maxG);
                        maxB = std::max((Rpp32f)srcPtrTemp[2], maxB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m256 result;
                reduce_max_float24_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

                maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
                maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
                maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);
#endif
            }
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = (Rpp16f)maxR;
			maxArr[maxArrIndex + 1] = (Rpp16f)maxG;
			maxArr[maxArrIndex + 2] = (Rpp16f)maxB;
			maxArr[maxArrIndex + 3] = (Rpp16f)maxC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_max_i8_i8_host(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *maxArr,
                                Rpp32u maxArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        // Tensor max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8s max = INT8_MIN;
            Rpp8s resultAvx[16];

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256i pMax = _mm256_set1_epi8(INT8_MIN);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_load_si256((__m256i *)srcPtrTemp);
                        pMax = _mm256_max_epi8(p1, pMax); //compare and store max of 32 values into global max

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        max = std::max(*srcPtrTemp++, max);
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_i32_host(&pMax, &result);
                rpp_simd_store(rpp_store16_i8, resultAvx, &result);

                max = std::max(resultAvx[0], max);
#endif
            maxArr[batchCount] = max;
        }
        // Tensor max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp8s maxC = INT8_MIN, maxR = INT8_MIN, maxG = INT8_MIN, maxB = INT8_MIN;
            Rpp8s resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
                __m256i pMaxR = _mm256_set1_epi8(INT8_MIN);
                __m256i pMaxG = pMaxR;
                __m256i pMaxB = pMaxR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256i p[3];
                        rpp_simd_load(rpp_load96_i8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                        compute_max_i96_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        maxR = std::max(*srcPtrTempR++, maxR);
                        maxG = std::max(*srcPtrTempG++, maxG);
                        maxB = std::max(*srcPtrTempB++, maxB);
                    }
                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_i96_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store16_i8, resultAvx, &result);

                maxR = std::max(resultAvx[0], maxR);
                maxG = std::max(resultAvx[1], maxG);
                maxB = std::max(resultAvx[2], maxB);
#endif
            }
            maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
            maxArr[maxArrIndex + 1] = maxG;
            maxArr[maxArrIndex + 2] = maxB;
            maxArr[maxArrIndex + 3] = maxC;
        }

        // Tensor max 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u maxArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8s maxC = INT8_MIN, maxR = INT8_MIN, maxG = INT8_MIN, maxB = INT8_MIN;
            Rpp8s resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow;
                srcPtrRow = srcPtrChannel;

                __m128i pMaxR = _mm_set1_epi8(INT8_MIN);
                __m128i pMaxG = pMaxR;
                __m128i pMaxB = pMaxR;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i p[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp, p);
                        compute_max_i48_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        maxR = std::max(srcPtrTemp[0], maxR);
                        maxG = std::max(srcPtrTemp[1], maxG);
                        maxB = std::max(srcPtrTemp[2], maxB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_max_i48_host(&pMaxR, &pMaxG, &pMaxB, &result);
                rpp_simd_store(rpp_store16_i8, resultAvx, &result);

                maxR = std::max(resultAvx[0], maxR);
                maxG = std::max(resultAvx[1], maxG);
                maxB = std::max(resultAvx[2], maxB);
#endif
            }
			maxC = std::max(std::max(maxR, maxG), maxB);
            maxArr[maxArrIndex] = maxR;
			maxArr[maxArrIndex + 1] = maxG;
			maxArr[maxArrIndex + 2] = maxB;
			maxArr[maxArrIndex + 3] = maxC;
        }
    }
    return RPP_SUCCESS;
}

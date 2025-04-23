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
#include "rpp_cpu_simd_math.hpp"

RppStatus phase_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f multiplier = RPP_255_OVER_1PT57;

        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        __m256 pMul = _mm256_set1_ps(multiplier);
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
#endif

        // Phase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[1] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[2] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[3] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[3], p2[3]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[4] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[4], p2[4]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[5] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[5], p2[5]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += vectorIncrement;
                    srcPtr2Temp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(srcPtr1Temp[0]) / static_cast<Rpp32f>(srcPtr2Temp[0])) * multiplier));
                    *dstPtrTempG++ = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(srcPtr1Temp[1]) / static_cast<Rpp32f>(srcPtr2Temp[1])) * multiplier));
                    *dstPtrTempB++ = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(srcPtr1Temp[2]) / static_cast<Rpp32f>(srcPtr2Temp[2])) * multiplier));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Phase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[1] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[2] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[3] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[3], p2[3]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[4] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[4], p2[4]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[5] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[5], p2[5]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += vectorIncrementPerChannel;
                    srcPtr1TempG += vectorIncrementPerChannel;
                    srcPtr1TempB += vectorIncrementPerChannel;
                    srcPtr2TempR += vectorIncrementPerChannel;
                    srcPtr2TempG += vectorIncrementPerChannel;
                    srcPtr2TempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(*srcPtr1TempR) / static_cast<Rpp32f>(*srcPtr2TempR)) * multiplier));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(*srcPtr1TempG) / static_cast<Rpp32f>(*srcPtr2TempG)) * multiplier));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(*srcPtr1TempB) / static_cast<Rpp32f>(*srcPtr2TempB)) * multiplier));

                    srcPtr1TempR++;
                    srcPtr1TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempR++;
                    srcPtr2TempG++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Phase without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
#if __AVX2__
            alignedLength = bufferLength & ~15;
#endif

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p1[2], p2[2];

                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                        p1[1] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += vectorIncrementPerChannel;
                        srcPtr2Temp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(atan(static_cast<Rpp32f>(*srcPtr1Temp) / static_cast<Rpp32f>(*srcPtr2Temp)) * multiplier));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus phase_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f multiplier = ONE_OVER_1PT57;

        Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        __m256 pMul = _mm256_set1_ps(multiplier);
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#endif

        // Phase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[3], p2[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                    p1[1] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul));    // phase computation
                    p1[2] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul));    // phase computation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += vectorIncrement;
                    srcPtr2Temp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = RPPPIXELCHECKF32(atan2f(srcPtr1Temp[0] , srcPtr2Temp[0]) * multiplier);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(atan2f(srcPtr1Temp[1] , srcPtr2Temp[1]) * multiplier);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(atan2f(srcPtr1Temp[2] , srcPtr2Temp[2]) * multiplier);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Phase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p1[3], p2[3];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                    p1[1] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul));    // phase computation
                    p1[2] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul));    // phase computation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += vectorIncrementPerChannel;
                    srcPtr1TempG += vectorIncrementPerChannel;
                    srcPtr1TempB += vectorIncrementPerChannel;
                    srcPtr2TempR += vectorIncrementPerChannel;
                    srcPtr2TempG += vectorIncrementPerChannel;
                    srcPtr2TempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(atan2f(*srcPtr1TempR , *srcPtr2TempR) * multiplier);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(atan2f(*srcPtr1TempG , *srcPtr2TempG) * multiplier);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(atan2f(*srcPtr1TempB , *srcPtr2TempB) * multiplier);

                    srcPtr1TempR++;
                    srcPtr1TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempR++;
                    srcPtr2TempG++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Phase without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#endif

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p1[1], p2[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += vectorIncrementPerChannel;
                        srcPtr2Temp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = RPPPIXELCHECKF32(atan2f(*srcPtr1Temp , *srcPtr2Temp) * multiplier);

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus phase_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f multiplier = ONE_OVER_1PT57;

        Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        __m256 pMul = _mm256_set1_ps(multiplier);
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#endif

        // Phase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtr1Temp_ps[24], srcPtr2Temp_ps[24];

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                    {
                        srcPtr1Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr1Temp[cnt]);
                        srcPtr2Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr2Temp[cnt]);
                    }

                    __m256 p1[3], p2[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp_ps, p1);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp_ps, p2);    // simd loads
                    p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                    p1[1] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul));    // phase computation
                    p1[2] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul));    // phase computation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += vectorIncrement;
                    srcPtr2Temp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(srcPtr1Temp[0] / srcPtr2Temp[0]) * multiplier));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(srcPtr1Temp[1] / srcPtr2Temp[1]) * multiplier));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(srcPtr1Temp[2] / srcPtr2Temp[2]) * multiplier));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Phase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtr1Temp_ps[24], srcPtr2Temp_ps[24];

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtr1Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr1TempR[cnt]);
                        srcPtr1Temp_ps[cnt + 8] = static_cast<Rpp32f>(srcPtr1TempG[cnt]);
                        srcPtr1Temp_ps[cnt + 16] = static_cast<Rpp32f>(srcPtr1TempB[cnt]);

                        srcPtr2Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr2TempR[cnt]);
                        srcPtr2Temp_ps[cnt + 8] = static_cast<Rpp32f>(srcPtr2TempG[cnt]);
                        srcPtr2Temp_ps[cnt + 16] = static_cast<Rpp32f>(srcPtr2TempB[cnt]);
                    }

                    __m256 p1[4], p2[4];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1Temp_ps, srcPtr1Temp_ps + 8, srcPtr1Temp_ps + 16, p1);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2Temp_ps, srcPtr2Temp_ps + 8, srcPtr2Temp_ps + 16, p2);    // simd loads
                    p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                    p1[1] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul));    // phase computation
                    p1[2] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul));    // phase computation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += vectorIncrementPerChannel;
                    srcPtr1TempG += vectorIncrementPerChannel;
                    srcPtr1TempB += vectorIncrementPerChannel;
                    srcPtr2TempR += vectorIncrementPerChannel;
                    srcPtr2TempG += vectorIncrementPerChannel;
                    srcPtr2TempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(*srcPtr1TempR / *srcPtr2TempR) * multiplier));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(*srcPtr1TempG / *srcPtr2TempG) * multiplier));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(*srcPtr1TempB / *srcPtr2TempB) * multiplier));

                    srcPtr1TempR++;
                    srcPtr1TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempR++;
                    srcPtr2TempG++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Phase without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#endif

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        Rpp32f srcPtr1Temp_ps[8], srcPtr2Temp_ps[8];

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtr1Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr1Temp[cnt]);
                            srcPtr2Temp_ps[cnt] = static_cast<Rpp32f>(srcPtr2Temp[cnt]);
                        }

                        __m256 p1[1], p2[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr1Temp_ps, p1);    // simd loads
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp_ps, p2);    // simd loads
                        p1[0] = rpp_pixel_check_0to1_avx(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul));    // phase computation
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += vectorIncrementPerChannel;
                        srcPtr2Temp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(atan(*srcPtr1Temp / *srcPtr2Temp) * multiplier));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus phase_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f multiplier = RPP_255_OVER_1PT57;

        Rpp8s *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        __m256 pMul = _mm256_set1_ps(multiplier);
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
#endif

        // Phase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[1] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[2] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[3] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[3], p2[3]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[4] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[4], p2[4]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[5] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[5], p2[5]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += vectorIncrement;
                    srcPtr2Temp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(srcPtr1Temp[0] + 128) / static_cast<Rpp32f>(srcPtr2Temp[0] + 128)) * multiplier) -  128));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(srcPtr1Temp[1] + 128) / static_cast<Rpp32f>(srcPtr2Temp[1] + 128)) * multiplier) -  128));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(srcPtr1Temp[2] + 128) / static_cast<Rpp32f>(srcPtr2Temp[2] + 128)) * multiplier) -  128));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Phase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[1] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[2] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[2], p2[2]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[3] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[3], p2[3]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[4] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[4], p2[4]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    p1[5] = _mm256_round_ps(_mm256_mul_ps(atan2_ps(p1[5], p2[5]), pMul), _MM_FROUND_TO_ZERO);    // phase computation
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += vectorIncrementPerChannel;
                    srcPtr1TempG += vectorIncrementPerChannel;
                    srcPtr1TempB += vectorIncrementPerChannel;
                    srcPtr2TempR += vectorIncrementPerChannel;
                    srcPtr2TempG += vectorIncrementPerChannel;
                    srcPtr2TempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(*srcPtr1TempR + 128) / static_cast<Rpp32f>(*srcPtr2TempR + 128)) * multiplier) -  128));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(*srcPtr1TempG + 128) / static_cast<Rpp32f>(*srcPtr2TempG + 128)) * multiplier) -  128));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(*srcPtr1TempB + 128) / static_cast<Rpp32f>(*srcPtr2TempB + 128)) * multiplier) -  128));

                    srcPtr1TempR++;
                    srcPtr1TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempR++;
                    srcPtr2TempG++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Phase without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
#if __AVX2__
            alignedLength = bufferLength & ~15;
#endif

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p1[2], p2[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_mul_ps(atan2_ps(p1[0], p2[0]), pMul);    // phase computation
                        p1[1] = _mm256_mul_ps(atan2_ps(p1[1], p2[1]), pMul);    // phase computation
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += vectorIncrementPerChannel;
                        srcPtr2Temp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((atan(static_cast<Rpp32f>(*srcPtr1Temp +  128) / static_cast<Rpp32f>(*srcPtr2Temp+  128)) * multiplier) -  128));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

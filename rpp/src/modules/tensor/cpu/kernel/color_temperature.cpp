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

inline void compute_color_temperature_48_host(__m256 *p, __m256 pAdj)
{
    p[0] = _mm256_add_ps(p[0], pAdj);    // color_temperature adjustment Rs
    p[1] = _mm256_add_ps(p[1], pAdj);    // color_temperature adjustment Rs
    // no color_temperature adjustment Gs
    p[4] = _mm256_sub_ps(p[4], pAdj);    // color_temperature adjustment Bs
    p[5] = _mm256_sub_ps(p[5], pAdj);    // color_temperature adjustment Bs
}

inline void compute_color_temperature_24_host(__m256 *p, __m256 pAdj)
{
    p[0] = _mm256_add_ps(p[0], pAdj);    // color_temperature adjustment Rs
    // no color_temperature adjustment Gs
    p[2] = _mm256_sub_ps(p[2], pAdj);    // color_temperature adjustment Bs
}

RppStatus color_temperature_u8_u8_host_tensor(Rpp8u *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp8u *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32s *adjustmentValueTensor,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f adjustmentValue = adjustmentValueTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m256 pAdj = _mm256_set1_ps(adjustmentValue);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Temperature with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[0] + adjustmentValue);
                    *dstPtrTempG++ = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[1]);
                    *dstPtrTempB++ = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(*srcPtrTempR + adjustmentValue);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(*srcPtrTempG);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(*srcPtrTempB - adjustmentValue);

                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[0] + adjustmentValue);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[1]);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp8u) RPPPIXELCHECK(*srcPtrTempR + adjustmentValue);
                    *dstPtrTempG++ = (Rpp8u) RPPPIXELCHECK(*srcPtrTempG);
                    *dstPtrTempB++ = (Rpp8u) RPPPIXELCHECK(*srcPtrTempB - adjustmentValue);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_temperature_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32s *adjustmentValueTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f adjustmentValue = adjustmentValueTensor[batchCount] * ONE_OVER_255;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m256 pAdj = _mm256_set1_ps(adjustmentValue);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Temperature with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = RPPPIXELCHECKF32(srcPtrTemp[0] + adjustmentValue);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(srcPtrTemp[1]);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(*srcPtrTempR + adjustmentValue);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(*srcPtrTempG);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(*srcPtrTempB - adjustmentValue);

                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 24;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(srcPtrTemp[0] + adjustmentValue);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(srcPtrTemp[1]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = RPPPIXELCHECKF32(*srcPtrTempR + adjustmentValue);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(*srcPtrTempG);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(*srcPtrTempB - adjustmentValue);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_temperature_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp16f *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32s *adjustmentValueTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f adjustmentValue = adjustmentValueTensor[batchCount] * ONE_OVER_255;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m256 pAdj = _mm256_set1_ps(adjustmentValue);

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u vectorIncrement = 24;

        // Color Temperature with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[0] + adjustmentValue);
                    *dstPtrTempG++ = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[1]);
                    *dstPtrTempB++ = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];

                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }

                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 24; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempR + adjustmentValue);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempG);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempB - adjustmentValue);

                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[25];

                    for(int cnt = 0; cnt < 24; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 24; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                    srcPtrTemp += 24;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[0] + adjustmentValue);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[1]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }

                    __m256 p[3];

                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_color_temperature_24_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempR + adjustmentValue);
                    *dstPtrTempG++ = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempG);
                    *dstPtrTempB++ = (Rpp16f) RPPPIXELCHECKF32(*srcPtrTempB - adjustmentValue);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_temperature_i8_i8_host_tensor(Rpp8s *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp8s *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32s *adjustmentValueTensor,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f adjustmentValue = adjustmentValueTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m256 pAdj = _mm256_set1_ps(adjustmentValue);

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Temperature with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[0] + adjustmentValue);
                    *dstPtrTempG++ = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[1]);
                    *dstPtrTempB++ = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempR + adjustmentValue);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempG);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempB - adjustmentValue);

                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[0] + adjustmentValue);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[1]);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(srcPtrTemp[2] - adjustmentValue);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Temperature with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_temperature_48_host(p, pAdj);    // color_temperature adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempR + adjustmentValue);
                    *dstPtrTempG++ = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempG);
                    *dstPtrTempB++ = (Rpp8s) RPPPIXELCHECKI8(*srcPtrTempB - adjustmentValue);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

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

RppStatus blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
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

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtr1Temp[0]) - (Rpp32f) (srcPtr2Temp[0])) * alpha) + (Rpp32f) (srcPtr2Temp[0])));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtr1Temp[1]) - (Rpp32f) (srcPtr2Temp[1])) * alpha) + (Rpp32f) (srcPtr2Temp[1])));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtr1Temp[2]) - (Rpp32f) (srcPtr2Temp[2])) * alpha) + (Rpp32f) (srcPtr2Temp[2])));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtr1TempR) - (Rpp32f) (*srcPtr2TempR)) * alpha) + (Rpp32f) (*srcPtr2TempR)));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtr1TempG) - (Rpp32f) (*srcPtr2TempG)) * alpha) + (Rpp32f) (*srcPtr2TempG)));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtr1TempB) - (Rpp32f) (*srcPtr2TempB)) * alpha) + (Rpp32f) (*srcPtr2TempB)));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
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

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128 p1[4], p2[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                        p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                        p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtr1Temp) - (Rpp32f) (*srcPtr2Temp)) * alpha) + (Rpp32f) (*srcPtr2Temp)));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
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

RppStatus blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
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

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp32f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32((srcPtr1Temp[0] - srcPtr2Temp[0]) * alpha + srcPtr2Temp[0]);
                    *dstPtrTempG = RPPPIXELCHECKF32((srcPtr1Temp[1] - srcPtr2Temp[1]) * alpha + srcPtr2Temp[1]);
                    *dstPtrTempB = RPPPIXELCHECKF32((srcPtr1Temp[2] - srcPtr2Temp[2]) * alpha + srcPtr2Temp[2]);

                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
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

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 4;
                    srcPtr1TempG += 4;
                    srcPtr1TempB += 4;
                    srcPtr2TempR += 4;
                    srcPtr2TempG += 4;
                    srcPtr2TempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtr1TempR - *srcPtr2TempR) * alpha + *srcPtr2TempR);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtr1TempG - *srcPtr2TempG) * alpha + *srcPtr2TempG);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtr1TempB - *srcPtr2TempB) * alpha + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
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

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                    {
                        __m128 p1[1], p2[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += 4;
                        srcPtr2Temp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtr1Temp - *srcPtr2Temp) * alpha + *srcPtr2Temp);

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
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

RppStatus blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
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

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp16f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    Rpp32f srcPtr1Temp_ps[12], srcPtr2Temp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtr1Temp_ps + cnt) = (Rpp32f) *(srcPtr1Temp + cnt);
                        *(srcPtr2Temp_ps + cnt) = (Rpp32f) *(srcPtr2Temp + cnt);
                    }

                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr1Temp_ps, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp_ps, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p1);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[0] - srcPtr2Temp[0]) * alpha + srcPtr2Temp[0]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[1] - srcPtr2Temp[1]) * alpha + srcPtr2Temp[1]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[2] - srcPtr2Temp[2]) * alpha + srcPtr2Temp[2]);

                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
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

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp32f srcPtr1Temp_ps[12], srcPtr2Temp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtr1Temp_ps + cnt) = (Rpp32f) *(srcPtr1TempR + cnt);
                        *(srcPtr1Temp_ps + 4 + cnt) = (Rpp32f) *(srcPtr1TempG + cnt);
                        *(srcPtr1Temp_ps + 8 + cnt) = (Rpp32f) *(srcPtr1TempB + cnt);

                        *(srcPtr2Temp_ps + cnt) = (Rpp32f) *(srcPtr2TempR + cnt);
                        *(srcPtr2Temp_ps + 4 + cnt) = (Rpp32f) *(srcPtr2TempG + cnt);
                        *(srcPtr2Temp_ps + 8 + cnt) = (Rpp32f) *(srcPtr2TempB + cnt);
                    }

                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr1Temp_ps, srcPtr1Temp_ps + 4, srcPtr1Temp_ps + 8, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2Temp_ps, srcPtr2Temp_ps + 4, srcPtr2Temp_ps + 8, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p1);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtr1TempR += 4;
                    srcPtr1TempG += 4;
                    srcPtr1TempB += 4;
                    srcPtr2TempR += 4;
                    srcPtr2TempG += 4;
                    srcPtr2TempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempR - *srcPtr2TempR) * alpha + *srcPtr2TempR);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempG - *srcPtr2TempG) * alpha + *srcPtr2TempG);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempB - *srcPtr2TempB) * alpha + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
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

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                    {
                        Rpp32f srcPtr1Temp_ps[4], srcPtr2Temp_ps[4], dstPtrTemp_ps[4];

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtr1Temp_ps + cnt) = (Rpp16f) *(srcPtr1Temp + cnt);
                            *(srcPtr2Temp_ps + cnt) = (Rpp16f) *(srcPtr2Temp + cnt);
                        }

                        __m128 p1[1], p2[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr1Temp_ps, p1);    // simd loads
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr2Temp_ps, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p1);    // simd stores

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }

                        srcPtr1Temp += 4;
                        srcPtr2Temp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtr1Temp - (Rpp32f)*srcPtr2Temp) * alpha + (Rpp32f)*srcPtr2Temp);

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
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

RppStatus blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
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

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp8s *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp8s *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[0]) - (Rpp32f) (srcPtr2Temp[0])) * alpha) + (Rpp32f) (srcPtr2Temp[0]));
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[1]) - (Rpp32f) (srcPtr2Temp[1])) * alpha) + (Rpp32f) (srcPtr2Temp[1]));
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[2]) - (Rpp32f) (srcPtr2Temp[2])) * alpha) + (Rpp32f) (srcPtr2Temp[2]));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempR) - (Rpp32f) (*srcPtr2TempR)) * alpha) + (Rpp32f) (*srcPtr2TempR));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempG) - (Rpp32f) (*srcPtr2TempG)) * alpha) + (Rpp32f) (*srcPtr2TempG));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempB) - (Rpp32f) (*srcPtr2TempB)) * alpha) + (Rpp32f) (*srcPtr2TempB));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
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

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128 p1[4], p2[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                        p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                        p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1Temp) - (Rpp32f) (*srcPtr2Temp)) * alpha) + (Rpp32f) (*srcPtr2Temp));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
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

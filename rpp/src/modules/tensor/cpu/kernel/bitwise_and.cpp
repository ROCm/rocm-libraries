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

/* bitwiseAND is logical operation only on U8 types.*/

RppStatus bitwise_and_u8_u8_host_tensor(Rpp8u *srcPtr1,
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

        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Bitwise AND with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m128i p1[3], p2[3];

                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_and_si128(p1[0], p2[0]);    // bitwise_and computation
                    p1[1] = _mm_and_si128(p1[1], p2[1]);    // bitwise_and computation
                    p1[2] = _mm_and_si128(p1[2], p2[2]);    // bitwise_and computation
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += vectorIncrement;
                    srcPtr2Temp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }

                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = srcPtr1Temp[0] & srcPtr2Temp[0];
                    *dstPtrTempG++ = srcPtr1Temp[1] & srcPtr2Temp[1];
                    *dstPtrTempB++ = srcPtr1Temp[2] & srcPtr2Temp[2];

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

        // Bitwise AND with fused output-layout toggle (NCHW -> NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i p1[3], p2[3];

                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_and_si128(p1[0], p2[0]);    // bitwise_and computation
                    p1[1] = _mm_and_si128(p1[1], p2[1]);    // bitwise_and computation
                    p1[2] = _mm_and_si128(p1[2], p2[2]);    // bitwise_and computation
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += vectorIncrementPerChannel;
                    srcPtr1TempG += vectorIncrementPerChannel;
                    srcPtr1TempB += vectorIncrementPerChannel;
                    srcPtr2TempR += vectorIncrementPerChannel;
                    srcPtr2TempG += vectorIncrementPerChannel;
                    srcPtr2TempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtr1TempR & *srcPtr2TempR;
                    dstPtrTemp[1] = *srcPtr1TempG & *srcPtr2TempG;
                    dstPtrTemp[2] = *srcPtr1TempB & *srcPtr2TempB;

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

        // Bitwise AND without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i p1, p2;

                        p1 = _mm_loadu_si128((__m128i *)srcPtr1Temp);   // simd loads
                        p2 = _mm_loadu_si128((__m128i *)srcPtr2Temp);   // simd loads
                        p1 = _mm_and_si128(p1, p2);    // bitwise_and computation
                        _mm_storeu_si128((__m128i *)dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += vectorIncrementPerChannel;
                        srcPtr2Temp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = *srcPtr1Temp & *srcPtr2Temp;

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

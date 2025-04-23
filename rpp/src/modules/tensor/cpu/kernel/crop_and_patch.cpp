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

RppStatus crop_and_patch_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                           Rpp8u *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
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
        RpptROIPtr roiPtrInput = &roiTensorPtrDst[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        RpptROIPtr cropRoi = &cropRoiTensor[batchCount];
        RpptROIPtr patchRoi = &patchRoiTensor[batchCount];
        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u cropBufferLength = cropRoi->xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength1 = patchRoi->xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength2 = (roi.xywhROI.roiWidth - (patchRoi->xywhROI.xy.x + cropRoi->xywhROI.roiWidth)) * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (cropRoi->xywhROI.xy.y * srcDescPtr->strides.hStride) + (cropRoi->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u cropAlignedLength = (cropBufferLength / 48) * 48;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 48) * 48;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 48) * 48;

            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr1Temp = srcPtr1Row;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr1Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr1Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr1Temp[0];
                        *dstPtrTempG++ = srcPtr1Temp[1];
                        *dstPtrTempB++ = srcPtr1Temp[2];
                        srcPtr1Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2Temp += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u cropAlignedLength = (cropBufferLength / 48) * 48;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 48) * 48;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 48) * 48;

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
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr1TempR = srcPtr1RowR;
                    srcPtr1TempG = srcPtr1RowG;
                    srcPtr1TempB = srcPtr1RowB;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr1TempR += vectorIncrementPerChannel;
                        srcPtr1TempG += vectorIncrementPerChannel;
                        srcPtr1TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr1TempR++;
                        dstPtrTemp[1] = *srcPtr1TempG++;
                        dstPtrTemp[2] = *srcPtr1TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2TempR += cropBufferLength;
                    srcPtr2TempG += cropBufferLength;
                    srcPtr2TempB += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                    srcPtr1RowR += srcDescPtr->strides.hStride;
                    srcPtr1RowG += srcDescPtr->strides.hStride;
                    srcPtr1RowB += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                }
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrRowTemp = dstPtrRow;
                Rpp8u *srcPtr1RowTemp = srcPtr1Row;
                Rpp8u *srcPtr2RowTemp = srcPtr2Row;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1);
                    srcPtr2RowTemp += patchBufferLength1;
                    dstPtrRowTemp += patchBufferLength1;
                    memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLength);
                    srcPtr2RowTemp += cropBufferLength;
                    dstPtrRowTemp += cropBufferLength;
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2);
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, bufferLength);
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *dstPtrRowTemp = dstPtrRow;
                    Rpp8u *srcPtr1RowTemp = srcPtr1Row;
                    Rpp8u *srcPtr2RowTemp = srcPtr2Row;
                    if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                    {
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1);
                        srcPtr2RowTemp += patchBufferLength1;
                        dstPtrRowTemp += patchBufferLength1;
                        memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLength);
                        srcPtr2RowTemp += cropBufferLength;
                        dstPtrRowTemp += cropBufferLength;
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2);
                        srcPtr1Row += srcDescPtr->strides.hStride;
                    }
                    else
                    {
                        memcpy(dstPtrRow, srcPtr2RowTemp, bufferLength);
                    }
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

RppStatus crop_and_patch_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                             Rpp32f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
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
        RpptROIPtr roiPtrInput = &roiTensorPtrDst[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        RpptROIPtr cropRoi = &cropRoiTensor[batchCount];
        RpptROIPtr patchRoi = &patchRoiTensor[batchCount];
        Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u cropBufferLength = cropRoi->xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength1 = patchRoi->xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength2 = (roi.xywhROI.roiWidth - (patchRoi->xywhROI.xy.x + cropRoi->xywhROI.roiWidth)) * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        Rpp32f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (cropRoi->xywhROI.xy.y * srcDescPtr->strides.hStride) + (cropRoi->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;
            Rpp32u cropAlignedLength = (cropBufferLength / 12) * 12;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 12) * 12;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 12) * 12;

            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrement)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr1Temp = srcPtr1Row;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr1Temp, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                        srcPtr1Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr1Temp[0];
                        *dstPtrTempG++ = srcPtr1Temp[1];
                        *dstPtrTempB++ = srcPtr1Temp[2];
                        srcPtr1Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2Temp += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrement)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;
            Rpp32u cropAlignedLength = (cropBufferLength / 12) * 12;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 12) * 12;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 12) * 12;

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
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount+=4)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr1TempR = srcPtr1RowR;
                    srcPtr1TempG = srcPtr1RowG;
                    srcPtr1TempB = srcPtr1RowB;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount+=4)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
                        srcPtr1TempR += vectorIncrementPerChannel;
                        srcPtr1TempG += vectorIncrementPerChannel;
                        srcPtr1TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr1TempR++;
                        dstPtrTemp[1] = *srcPtr1TempG++;
                        dstPtrTemp[2] = *srcPtr1TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2TempR += cropBufferLength;
                    srcPtr2TempG += cropBufferLength;
                    srcPtr2TempB += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount+=4)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                    srcPtr1RowR += srcDescPtr->strides.hStride;
                    srcPtr1RowG += srcDescPtr->strides.hStride;
                    srcPtr1RowB += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                }
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u bufferLengthInBytes = bufferLength * sizeof(Rpp32f);
            Rpp32u cropBufferLengthInBytes = cropBufferLength * sizeof(Rpp32f);
            Rpp32u patchBufferLength1InBytes = patchBufferLength1 * sizeof(Rpp32f);
            Rpp32u patchBufferLength2InBytes = patchBufferLength2 * sizeof(Rpp32f);

            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrRowTemp = dstPtrRow;
                Rpp32f *srcPtr1RowTemp = srcPtr1Row;
                Rpp32f *srcPtr2RowTemp = srcPtr2Row;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1InBytes);
                    srcPtr2RowTemp += patchBufferLength1;
                    dstPtrRowTemp += patchBufferLength1;
                    memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLengthInBytes);
                    srcPtr2RowTemp += cropBufferLength;
                    dstPtrRowTemp += cropBufferLength;
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2InBytes);
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, bufferLengthInBytes);
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32u bufferLengthInBytes = bufferLength * sizeof(Rpp32f);
            Rpp32u cropBufferLengthInBytes = cropBufferLength * sizeof(Rpp32f);
            Rpp32u patchBufferLength1InBytes = patchBufferLength1 * sizeof(Rpp32f);
            Rpp32u patchBufferLength2InBytes = patchBufferLength2 * sizeof(Rpp32f);
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *dstPtrRowTemp = dstPtrRow;
                    Rpp32f *srcPtr1RowTemp = srcPtr1Row;
                    Rpp32f *srcPtr2RowTemp = srcPtr2Row;
                    if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                    {
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1InBytes);
                        srcPtr2RowTemp += patchBufferLength1;
                        dstPtrRowTemp += patchBufferLength1;
                        memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLengthInBytes);
                        srcPtr2RowTemp += cropBufferLength;
                        dstPtrRowTemp += cropBufferLength;
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2InBytes);
                        srcPtr1Row += srcDescPtr->strides.hStride;
                    }
                    else
                    {
                        memcpy(dstPtrRow, srcPtr2RowTemp, bufferLengthInBytes);
                    }
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

RppStatus crop_and_patch_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                             Rpp16f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
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
        RpptROIPtr roiPtrInput = &roiTensorPtrDst[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        RpptROIPtr cropRoi = &cropRoiTensor[batchCount];
        RpptROIPtr patchRoi = &patchRoiTensor[batchCount];
        Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u cropBufferLength = cropRoi->xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength1 = patchRoi->xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength2 = (roi.xywhROI.roiWidth - (patchRoi->xywhROI.xy.x + cropRoi->xywhROI.roiWidth)) * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        Rpp16f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (cropRoi->xywhROI.xy.y * srcDescPtr->strides.hStride) + (cropRoi->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;
            Rpp32u cropAlignedLength = (cropBufferLength / 12) * 12;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 12) * 12;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 12) * 12;

            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2Temp + cnt);

                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                            *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                        }
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                    vectorLoopCount = 0;
                    srcPtr1Temp = srcPtr1Row;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr1Temp + cnt);

                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                            *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                        }
                        srcPtr1Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr1Temp[0];
                        *dstPtrTempG++ = srcPtr1Temp[1];
                        *dstPtrTempB++ = srcPtr1Temp[2];
                        srcPtr1Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2Temp += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2Temp + cnt);

                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                            *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                        }
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2Temp + cnt);

                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                            *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                        }
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;
            Rpp32u cropAlignedLength = (cropBufferLength / 12) * 12;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 12) * 12;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 12) * 12;

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
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;

                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2TempR + cnt);
                            *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtr2TempG + cnt);
                            *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtr2TempB + cnt);
                        }
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 4, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                    vectorLoopCount = 0;
                    srcPtr1TempR = srcPtr1RowR;
                    srcPtr1TempG = srcPtr1RowG;
                    srcPtr1TempB = srcPtr1RowB;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr1TempR + cnt);
                            *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtr1TempG + cnt);
                            *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtr1TempB + cnt);
                        }
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                        srcPtr1TempR += vectorIncrementPerChannel;
                        srcPtr1TempG += vectorIncrementPerChannel;
                        srcPtr1TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr1TempR++;
                        dstPtrTemp[1] = *srcPtr1TempG++;
                        dstPtrTemp[2] = *srcPtr1TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2TempR += cropBufferLength;
                    srcPtr2TempG += cropBufferLength;
                    srcPtr2TempB += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2TempR + cnt);
                            *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtr2TempG + cnt);
                            *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtr2TempB + cnt);
                        }
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                    srcPtr1RowR += srcDescPtr->strides.hStride;
                    srcPtr1RowG += srcDescPtr->strides.hStride;
                    srcPtr1RowB += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr2TempR + cnt);
                            *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtr2TempG + cnt);
                            *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtr2TempB + cnt);
                        }
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                        for(int cnt = 0; cnt < 12; cnt++)
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                }

                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u bufferLengthInBytes = bufferLength * sizeof(Rpp16f);
            Rpp32u cropBufferLengthInBytes = cropBufferLength * sizeof(Rpp16f);
            Rpp32u patchBufferLength1InBytes = patchBufferLength1 * sizeof(Rpp16f);
            Rpp32u patchBufferLength2InBytes = patchBufferLength2 * sizeof(Rpp16f);

            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrRowTemp = dstPtrRow;
                Rpp16f *srcPtr1RowTemp = srcPtr1Row;
                Rpp16f *srcPtr2RowTemp = srcPtr2Row;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1InBytes);
                    srcPtr2RowTemp += patchBufferLength1;
                    dstPtrRowTemp += patchBufferLength1;
                    memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLengthInBytes);
                    srcPtr2RowTemp += cropBufferLength;
                    dstPtrRowTemp += cropBufferLength;
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2InBytes);
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, bufferLengthInBytes);
                }

                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32u bufferLengthInBytes = bufferLength * sizeof(Rpp16f);
            Rpp32u cropBufferLengthInBytes = cropBufferLength * sizeof(Rpp16f);
            Rpp32u patchBufferLength1InBytes = patchBufferLength1 * sizeof(Rpp16f);
            Rpp32u patchBufferLength2InBytes = patchBufferLength2 * sizeof(Rpp16f);
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *dstPtrRowTemp = dstPtrRow;
                    Rpp16f *srcPtr1RowTemp = srcPtr1Row;
                    Rpp16f *srcPtr2RowTemp = srcPtr2Row;
                    if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                    {
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1InBytes);
                        srcPtr2RowTemp += patchBufferLength1;
                        dstPtrRowTemp += patchBufferLength1;
                        memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLengthInBytes);
                        srcPtr2RowTemp += cropBufferLength;
                        dstPtrRowTemp += cropBufferLength;
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2InBytes);
                        srcPtr1Row += srcDescPtr->strides.hStride;
                    }
                    else
                    {
                        memcpy(dstPtrRow, srcPtr2RowTemp, bufferLengthInBytes);
                    }
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

RppStatus crop_and_patch_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                           Rpp8s *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
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
        RpptROIPtr roiPtrInput = &roiTensorPtrDst[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        RpptROIPtr cropRoi = &cropRoiTensor[batchCount];
        RpptROIPtr patchRoi = &patchRoiTensor[batchCount];
        Rpp8s *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u cropBufferLength = cropRoi->xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength1 = patchRoi->xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u patchBufferLength2 = (roi.xywhROI.roiWidth - (patchRoi->xywhROI.xy.x + cropRoi->xywhROI.roiWidth)) * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        Rpp8s *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (cropRoi->xywhROI.xy.y * srcDescPtr->strides.hStride) + (cropRoi->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u cropAlignedLength = (cropBufferLength / 48) * 48;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 48) * 48;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 48) * 48;

            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;

                    }

                    vectorLoopCount = 0;
                    srcPtr1Temp = srcPtr1Row;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtr1Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr1Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr1Temp[0];
                        *dstPtrTempG++ = srcPtr1Temp[1];
                        *dstPtrTempB++ = srcPtr1Temp[2];
                        srcPtr1Temp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2Temp += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtr2Temp, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                        srcPtr2Temp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR++ = srcPtr2Temp[0];
                        *dstPtrTempG++ = srcPtr2Temp[1];
                        *dstPtrTempB++ = srcPtr2Temp[2];
                        srcPtr2Temp += 3;
                    }
                }

                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u cropAlignedLength = (cropBufferLength / 48) * 48;
            Rpp32u patchAlignedLength1 = (patchBufferLength1 / 48) * 48;
            Rpp32u patchAlignedLength2 = (patchBufferLength2 / 48) * 48;

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
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength1; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength1; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr1TempR = srcPtr1RowR;
                    srcPtr1TempG = srcPtr1RowG;
                    srcPtr1TempB = srcPtr1RowB;
#if __AVX2__
                    for (; vectorLoopCount < cropAlignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr1TempR += vectorIncrementPerChannel;
                        srcPtr1TempG += vectorIncrementPerChannel;
                        srcPtr1TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < cropBufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr1TempR++;
                        dstPtrTemp[1] = *srcPtr1TempG++;
                        dstPtrTemp[2] = *srcPtr1TempB++;
                        dstPtrTemp += 3;
                    }

                    vectorLoopCount = 0;
                    srcPtr2TempR += cropBufferLength;
                    srcPtr2TempG += cropBufferLength;
                    srcPtr2TempB += cropBufferLength;
#if __AVX2__
                    for(; vectorLoopCount < patchAlignedLength2; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < patchBufferLength2; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                    srcPtr1RowR += srcDescPtr->strides.hStride;
                    srcPtr1RowG += srcDescPtr->strides.hStride;
                    srcPtr1RowB += srcDescPtr->strides.hStride;
                }
                else
                {
#if __AVX2__
                    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m128i px[3];
                        rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, px);    // simd loads
                        rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                        srcPtr2TempR += vectorIncrementPerChannel;
                        srcPtr2TempG += vectorIncrementPerChannel;
                        srcPtr2TempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = *srcPtr2TempR++;
                        dstPtrTemp[1] = *srcPtr2TempG++;
                        dstPtrTemp[2] = *srcPtr2TempB++;
                        dstPtrTemp += 3;
                    }
                }
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrRowTemp = dstPtrRow;
                Rpp8s *srcPtr1RowTemp = srcPtr1Row;
                Rpp8s *srcPtr2RowTemp = srcPtr2Row;
                if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1);
                    srcPtr2RowTemp += patchBufferLength1;
                    dstPtrRowTemp += patchBufferLength1;
                    memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLength);
                    srcPtr2RowTemp += cropBufferLength;
                    dstPtrRowTemp += cropBufferLength;
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2);
                    srcPtr1Row += srcDescPtr->strides.hStride;
                }
                else
                {
                    memcpy(dstPtrRowTemp, srcPtr2RowTemp, bufferLength);
                }
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *dstPtrRowTemp = dstPtrRow;
                    Rpp8s *srcPtr1RowTemp = srcPtr1Row;
                    Rpp8s *srcPtr2RowTemp = srcPtr2Row;
                    if(i >= patchRoi->xywhROI.xy.y && i < (patchRoi->xywhROI.xy.y + cropRoi->xywhROI.roiHeight))
                    {
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength1);
                        srcPtr2RowTemp += patchBufferLength1;
                        dstPtrRowTemp += patchBufferLength1;
                        memcpy(dstPtrRowTemp, srcPtr1RowTemp, cropBufferLength);
                        srcPtr2RowTemp += cropBufferLength;
                        dstPtrRowTemp += cropBufferLength;
                        memcpy(dstPtrRowTemp, srcPtr2RowTemp, patchBufferLength2);
                        srcPtr1Row += srcDescPtr->strides.hStride;
                    }
                    else
                    {
                        memcpy(dstPtrRow, srcPtr2RowTemp, bufferLength);
                    }
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

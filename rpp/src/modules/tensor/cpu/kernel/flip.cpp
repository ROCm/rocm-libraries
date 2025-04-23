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

RppStatus flip_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
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

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        //Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src pointer location)
        Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
        Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

        //Initialize load functions with default values
        auto load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_avx;
        auto load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_avx;
        auto load16Fn = &rpp_load16_u8_to_f32_avx;

        //Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags enabled
        if(horizontalFlag == 1)
        {
            hFactor += (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_u8_to_f32_mirror_avx;
        }
        if(verticalFlag == 1)
        {
            vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
            hStrideSrcIncrement = -srcDescPtr->strides.hStride;
        }
        srcPtrChannel = srcPtrImage + vFactor + hFactor;

        //Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - 3) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];

                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);     // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[0]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[1]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];

                    rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += srcPtrIncrementPerChannel;
                    srcPtrTempG += srcPtrIncrementPerChannel;
                    srcPtrTempB += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }

                srcPtrTempR += hFlipFactorPerChannel;
                srcPtrTempG += hFlipFactorPerChannel;
                srcPtrTempB += hFlipFactorPerChannel;

                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempR));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempG));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempB));

                    srcPtrTempR += srcPtrIncrementPerPixel;
                    srcPtrTempG += srcPtrIncrementPerPixel;
                    srcPtrTempB += srcPtrIncrementPerPixel;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += hStrideSrcIncrement;;
                srcPtrRowG += hStrideSrcIncrement;
                srcPtrRowB += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[0]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[1]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[2]));
                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTemp += 3;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];
                        rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += srcPtrIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    srcPtrTemp += hFlipFactorPerChannel;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTemp));
                        srcPtrTemp += srcPtrIncrementPerPixel;
                        dstPtrTemp++;
                    }

                    srcPtrRow += hStrideSrcIncrement;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
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

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        //Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src pointer location)
        Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
        Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

        // Initialize load functions with default values
        auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
        auto load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_avx;
        auto load8Fn = &rpp_load8_f32_to_f32_avx;

        //Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags enabled
        if(horizontalFlag == 1)
        {
            hFactor += (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
            load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
            load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
        }
        if(verticalFlag == 1)
        {
            vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
            hStrideSrcIncrement = -srcDescPtr->strides.hStride;
        }
        srcPtrChannel = srcPtrImage + vFactor + hFactor;

        //Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp32f);
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];

                    rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);     // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = srcPtrTemp[0];
                    *dstPtrTempG = srcPtrTemp[1];
                    *dstPtrTempB = srcPtrTemp[2];

                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];

                    rpp_simd_load(load24FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += srcPtrIncrementPerChannel;
                    srcPtrTempG += srcPtrIncrementPerChannel;
                    srcPtrTempB += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }

                srcPtrTempR += hFlipFactorPerChannel;
                srcPtrTempG += hFlipFactorPerChannel;
                srcPtrTempB += hFlipFactorPerChannel;

                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTempR;
                    dstPtrTemp[1] = *srcPtrTempG;
                    dstPtrTemp[2] = *srcPtrTempB;

                    srcPtrTempR += srcPtrIncrementPerPixel;
                    srcPtrTempG += srcPtrIncrementPerPixel;
                    srcPtrTempB += srcPtrIncrementPerPixel;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += hStrideSrcIncrement;
                srcPtrRowG += hStrideSrcIncrement;
                srcPtrRowB += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = srcPtrTemp[0];
                    dstPtrTemp[1] = srcPtrTemp[1];
                    dstPtrTemp[2] = srcPtrTemp[2];
                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTemp += 3;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];
                        rpp_simd_load(load8Fn, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += srcPtrIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    srcPtrTemp += hFlipFactorPerChannel;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = *srcPtrTemp;
                        srcPtrTemp += srcPtrIncrementPerPixel;
                        dstPtrTemp++;
                    }

                    srcPtrRow += hStrideSrcIncrement;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
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

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        //Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src pointer location)
        Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
        Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

        // Initialize load functions with default values
        auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
        auto load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_avx;
        auto load8Fn = &rpp_load8_f32_to_f32_avx;

        //Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags enabled
        if(horizontalFlag == 1)
        {
            hFactor += (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
            load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
            load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
        }
        if(verticalFlag == 1)
        {
            vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
            hStrideSrcIncrement = -srcDescPtr->strides.hStride;
        }
        srcPtrChannel = srcPtrImage + vFactor + hFactor;

        //Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp16f);
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
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
                    __m256 p[6];

                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    rpp_simd_load(load24FnPkdPln, srcPtrTemp_ps, p);     // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);

                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];

                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }

                    rpp_simd_load(load24FnPlnPln, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += srcPtrIncrementPerChannel;
                    srcPtrTempG += srcPtrIncrementPerChannel;
                    srcPtrTempB += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }

                srcPtrTempR += hFlipFactorPerChannel;
                srcPtrTempG += hFlipFactorPerChannel;
                srcPtrTempB += hFlipFactorPerChannel;

                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB);

                    srcPtrTempR += srcPtrIncrementPerPixel;
                    srcPtrTempG += srcPtrIncrementPerPixel;
                    srcPtrTempB += srcPtrIncrementPerPixel;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += hStrideSrcIncrement;
                srcPtrRowG += hStrideSrcIncrement;
                srcPtrRowB += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];

                    Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    rpp_simd_load(load24FnPkdPln, srcPtrTemp_ps, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);
                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTemp += 3;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];

                        Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        rpp_simd_load(load8Fn, srcPtrTemp_ps, p);    // simd loads
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        srcPtrTemp += srcPtrIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    srcPtrTemp -= hFlipFactorPerChannel;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp);
                        srcPtrTemp += srcPtrIncrementPerPixel;
                        dstPtrTemp++;
                    }

                    srcPtrRow += hStrideSrcIncrement;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
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

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        //Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src pointer location)
        Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
        Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
        Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

        //Initialize load functions with default values
        auto load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_avx;
        auto load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_avx;
        auto load16Fn = &rpp_load16_i8_to_f32_avx;

        //Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags enabled
        if(horizontalFlag == 1)
        {
            hFactor += (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
        }
        if(verticalFlag == 1)
        {
            vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
            hStrideSrcIncrement = -srcDescPtr->strides.hStride;
        }
        srcPtrChannel = srcPtrImage + vFactor + hFactor;

        //Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];

                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);     // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];

                    rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += srcPtrIncrementPerChannel;
                    srcPtrTempG += srcPtrIncrementPerChannel;
                    srcPtrTempB += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }

                srcPtrTempR += hFlipFactorPerChannel;
                srcPtrTempG += hFlipFactorPerChannel;
                srcPtrTempB += hFlipFactorPerChannel;

                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempR));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempG));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempB));

                    srcPtrTempR += srcPtrIncrementPerPixel;
                    srcPtrTempG += srcPtrIncrementPerPixel;
                    srcPtrTempB += srcPtrIncrementPerPixel;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += hStrideSrcIncrement;;
                srcPtrRowG += hStrideSrcIncrement;
                srcPtrRowB += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += srcPtrIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                srcPtrTemp += hFlipFactor;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));
                    srcPtrTemp += srcPtrIncrementPerRGB;
                    dstPtrTemp += 3;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];
                        rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += srcPtrIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTemp));
                        srcPtrTemp += srcPtrIncrementPerPixel;
                        dstPtrTemp++;
                    }

                    srcPtrRow += hStrideSrcIncrement;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

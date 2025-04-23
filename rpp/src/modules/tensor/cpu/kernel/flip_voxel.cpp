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

RppStatus flip_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32u *horizontalTensor,
                                         Rpp32u *verticalTensor,
                                         Rpp32u *depthTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle)
{
    RpptROI3D roiDefault;
    if (srcGenericDescPtr->layout == RpptLayout::NCDHW)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2]};
    else if (srcGenericDescPtr->layout == RpptLayout::NDHWC)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2], (Rpp32s)srcGenericDescPtr->dims[1]};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];
        Rpp32u depthFlag = depthTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;

        // Initialize load functions with default values
        auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
        auto load8Fn = &rpp_load8_f32_to_f32_avx;

        // Update the load functions, horizontalFactor, verticalFactor and horizontalStrideSrcIncrement based on the flags enabled
        if (horizontalFlag == 1)
        {
            load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
            load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
        }

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        // Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - layoutParams.bufferMultiplier) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NCDHW -> NCDHW)
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp32u horizontalFactor = roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier;
            Rpp32u verticalFactor = roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3];
            Rpp32u depthFactor = roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2];
            Rpp32s horizontalStrideSrcIncrement = srcGenericDescPtr->strides[3];
            Rpp32s depthStrideIncrement = srcGenericDescPtr->strides[2];

            if (horizontalFlag)
                horizontalFactor += (roi.xyzwhdROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            if (verticalFlag)
            {
                verticalFactor += (roi.xyzwhdROI.roiHeight - 1) * srcGenericDescPtr->strides[3];
                horizontalStrideSrcIncrement = -srcGenericDescPtr->strides[3];
            }
            if (depthFlag)
            {
                depthFactor =  (roi.xyzwhdROI.roiDepth - 1) * srcGenericDescPtr->strides[2];
                depthStrideIncrement = -srcGenericDescPtr->strides[2];
            }
            srcPtrChannel = srcPtrImage + depthFactor + verticalFactor + horizontalFactor;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrDepth, *dstPtrDepth;
                srcPtrDepth = srcPtrChannel;
                dstPtrDepth = dstPtrChannel;
                for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
                {
                    Rpp32f *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrDepth;
                    dstPtrRow = dstPtrDepth;
                    for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                    {
                        Rpp32f *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;
                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[1];
                            rpp_simd_load(load8Fn, srcPtrTemp, p);    // simd loads
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                            srcPtrTemp += srcPtrIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
#endif
                        srcPtrTemp += hFlipFactorPerChannel;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = *srcPtrTemp;
                            srcPtrTemp += srcPtrIncrementPerPixel;
                        }
                        srcPtrRow += horizontalStrideSrcIncrement;
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += depthStrideIncrement;
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
                srcPtrChannel += srcGenericDescPtr->strides[1];
                dstPtrChannel += srcGenericDescPtr->strides[1];
            }
        }

        // flip without fused output-layout toggle (NDHWC -> NDHWC)
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        {
            Rpp32u horizontalFactor = roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier;
            Rpp32u verticalFactor = roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2];
            Rpp32u depthFactor = roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1];
            Rpp32s horizontalStrideSrcIncrement = srcGenericDescPtr->strides[2];
            Rpp32s depthStrideIncrement = srcGenericDescPtr->strides[1];

            if (horizontalFlag)
                horizontalFactor += (roi.xyzwhdROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            if (verticalFlag)
            {
                verticalFactor += (roi.xyzwhdROI.roiHeight - 1) * srcGenericDescPtr->strides[2];
                horizontalStrideSrcIncrement = -srcGenericDescPtr->strides[2];
            }
            if (depthFlag)
            {
                depthFactor =  (roi.xyzwhdROI.roiDepth - 1) * srcGenericDescPtr->strides[1];
                depthStrideIncrement = -srcGenericDescPtr->strides[1];
            }
            srcPtrChannel = srcPtrImage + depthFactor + verticalFactor + horizontalFactor;

            Rpp32f *srcPtrDepth = srcPtrChannel;
            Rpp32f *dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;
                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);  // simd stores
                        srcPtrTemp += srcPtrIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    srcPtrTemp += hFlipFactor;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = srcPtrTemp[0];
                        dstPtrTemp[1] = srcPtrTemp[1];
                        dstPtrTemp[2] = srcPtrTemp[2];
                        srcPtrTemp += srcPtrIncrementPerRGB;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += horizontalStrideSrcIncrement;
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += depthStrideIncrement;
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       Rpp32u *horizontalTensor,
                                       Rpp32u *verticalTensor,
                                       Rpp32u *depthTensor,
                                       RpptROI3DPtr roiGenericPtrSrc,
                                       RpptRoi3DType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI3D roiDefault;
    if (srcGenericDescPtr->layout == RpptLayout::NCDHW)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2]};
    else if (srcGenericDescPtr->layout == RpptLayout::NDHWC)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2], (Rpp32s)srcGenericDescPtr->dims[1]};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];
        Rpp32u depthFlag = depthTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;

        // Initialize load functions with default values
        auto load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_avx;
        auto load16Fn = &rpp_load16_u8_to_f32_avx;

        // Update the load functions, horizontalFactor, verticalFactor and horizontalStrideSrcIncrement based on the flags enabled
        if (horizontalFlag == 1)
        {
            load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_u8_to_f32_mirror_avx;
        }

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        // Compute constant increment, Decrement factors used in source pointer updation
        Rpp32s srcPtrIncrement = (horizontalFlag)? -vectorIncrement : vectorIncrement;
        Rpp32u hFlipFactor = (vectorIncrement - layoutParams.bufferMultiplier) * horizontalFlag;
        Rpp32s srcPtrIncrementPerChannel = (horizontalFlag)? -vectorIncrementPerChannel : vectorIncrementPerChannel;
        Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
        Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
        Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

        // flip without fused output-layout toggle (NCDHW -> NCDHW)
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp32u horizontalFactor = roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier;
            Rpp32u verticalFactor = roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3];
            Rpp32u depthFactor = roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2];
            Rpp32s horizontalStrideSrcIncrement = srcGenericDescPtr->strides[3];
            Rpp32s depthStrideIncrement = srcGenericDescPtr->strides[2];

            if (horizontalFlag)
                horizontalFactor += (roi.xyzwhdROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            if (verticalFlag)
            {
                verticalFactor += (roi.xyzwhdROI.roiHeight - 1) * srcGenericDescPtr->strides[3];
                horizontalStrideSrcIncrement = -srcGenericDescPtr->strides[3];
            }
            if (depthFlag)
            {
                depthFactor =  (roi.xyzwhdROI.roiDepth - 1) * srcGenericDescPtr->strides[2];
                depthStrideIncrement = -srcGenericDescPtr->strides[2];
            }
            srcPtrChannel = srcPtrImage + depthFactor + verticalFactor + horizontalFactor;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrDepth, *dstPtrDepth;
                srcPtrDepth = srcPtrChannel;
                dstPtrDepth = dstPtrChannel;
                for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
                {
                    Rpp8u *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrDepth;
                    dstPtrRow = dstPtrDepth;
                    for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                    {
                        Rpp8u *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;
                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];
                            rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);  // simd stores
                            srcPtrTemp += srcPtrIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
#endif
                        srcPtrTemp += hFlipFactorPerChannel;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = *srcPtrTemp;
                            srcPtrTemp += srcPtrIncrementPerPixel;
                        }
                        srcPtrRow += horizontalStrideSrcIncrement;
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += depthStrideIncrement;
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
                srcPtrChannel += srcGenericDescPtr->strides[1];
                dstPtrChannel += dstGenericDescPtr->strides[1];
            }
        }

        // flip without fused output-layout toggle (NDHWC -> NDHWC)
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        {
            Rpp32u horizontalFactor = roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier;
            Rpp32u verticalFactor = roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2];
            Rpp32u depthFactor = roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1];
            Rpp32s horizontalStrideSrcIncrement = srcGenericDescPtr->strides[2];
            Rpp32s depthStrideIncrement = srcGenericDescPtr->strides[1];

            if (horizontalFlag)
                horizontalFactor += (roi.xyzwhdROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
            if (verticalFlag)
            {
                verticalFactor += (roi.xyzwhdROI.roiHeight - 1) * srcGenericDescPtr->strides[2];
                horizontalStrideSrcIncrement = -srcGenericDescPtr->strides[2];
            }
            if (depthFlag)
            {
                depthFactor =  (roi.xyzwhdROI.roiDepth - 1) * srcGenericDescPtr->strides[1];
                depthStrideIncrement = -srcGenericDescPtr->strides[1];
            }
            srcPtrChannel = srcPtrImage + depthFactor + verticalFactor + horizontalFactor;

            Rpp8u *srcPtrDepth = srcPtrChannel;
            Rpp8u *dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;
                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {

                        __m256 p[6];
                        rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);  // simd stores
                        srcPtrTemp += srcPtrIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    srcPtrTemp += hFlipFactor;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = srcPtrTemp[0];
                        dstPtrTemp[1] = srcPtrTemp[1];
                        dstPtrTemp[2] = srcPtrTemp[2];
                        srcPtrTemp += srcPtrIncrementPerRGB;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += horizontalStrideSrcIncrement;
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += depthStrideIncrement;
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }
    }

    return RPP_SUCCESS;
}

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

inline void compute_add_16_host(__m256 *p, __m256 *pAddParam)
{
    p[0] = _mm256_add_ps(p[0], pAddParam[0]);    // add adjustment
    p[1] = _mm256_add_ps(p[1], pAddParam[0]);    // add adjustment
}

RppStatus add_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32f *addTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle)
{
    RpptROI3D roiDefault;
    if(srcGenericDescPtr->layout==RpptLayout::NCDHW)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2]};
    else if(srcGenericDescPtr->layout==RpptLayout::NDHWC)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2], (Rpp32s)srcGenericDescPtr->dims[1]};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32f addParam = addTensor[batchCount];
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement = 16;
        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
        __m256 pAddParam = _mm256_set1_ps(addParam);

        // Add without fused output-layout toggle (NCDHW -> NCDHW)
        if((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);

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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_add_16_host(p, &pAddParam);                         // add adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                            srcPtrTemp += vectorIncrement;
                            dstPtrTemp += vectorIncrement;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = *srcPtrTemp++ + addParam;
                        }
                        srcPtrRow += srcGenericDescPtr->strides[3];
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[2];
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
                srcPtrChannel += srcGenericDescPtr->strides[1];
                dstPtrChannel += srcGenericDescPtr->strides[1];
            }
        }
        // Add without fused output-layout toggle (NDHWC -> NDHWC)
        else if((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        {
            srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_add_16_host(p, &pAddParam);                         // add adjustment
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = *srcPtrTemp++ + addParam;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += srcGenericDescPtr->strides[1];
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }
    }

    return RPP_SUCCESS;
}

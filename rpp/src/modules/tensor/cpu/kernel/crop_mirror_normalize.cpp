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

inline void compute_cmn_48_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[2], pCMNParams[3]);
    p[3] = _mm256_fmadd_ps(p[3], pCMNParams[2], pCMNParams[3]);
    p[4] = _mm256_fmadd_ps(p[4], pCMNParams[4], pCMNParams[5]);
    p[5] = _mm256_fmadd_ps(p[5], pCMNParams[4], pCMNParams[5]);
}

inline void compute_cmn_48_rgb_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[0], pCMNParams[1]);
    p[3] = _mm256_fmadd_ps(p[3], pCMNParams[0], pCMNParams[1]);
    p[4] = _mm256_fmadd_ps(p[4], pCMNParams[0], pCMNParams[1]);
    p[5] = _mm256_fmadd_ps(p[5], pCMNParams[0], pCMNParams[1]);
    p[6] = _mm256_fmadd_ps(p[6], pCMNParams[0], pCMNParams[1]);
    p[7] = _mm256_fmadd_ps(p[7], pCMNParams[0], pCMNParams[1]);
}

inline void compute_cmn_24_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[2], pCMNParams[3]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[4], pCMNParams[5]);
}

inline void compute_cmn_16_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
}

inline void compute_cmn_8_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
}

RppStatus crop_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        for(int pos = 0; pos < numRegs; pos += 2)
        {
            pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
            pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
            cmnParamLoc++;
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     //simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]))));
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]))));
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]))));

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      //simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]))));
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]))));
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]))));

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempR)) * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempG)) * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempB)) * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]));

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempR)) * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempG)) * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempB)) * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]));

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     //simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]))));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]))));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]))));
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]))));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]))));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]))));
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTemp)) * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]));

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTemp)) * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]));
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        for(int pos = 0; pos < numRegs; pos += 2)
        {
            pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
            pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
            cmnParamLoc++;
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = ((srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = ((srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = ((srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = ((srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = ((srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = ((srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = ((*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = ((*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = ((*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = ((*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = ((*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = ((*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = ((srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = ((srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = ((srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = ((srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = ((srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = ((srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                            __m256 p[1];
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = ((*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[1];
                            rpp_simd_load(rpp_load8_f32_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = ((*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        for(int pos = 0; pos < numRegs; pos += 2)
        {
            pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
            pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
            cmnParamLoc++;
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);     //simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        Rpp32f srcPtrTemp_ps[24];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp_ps, p);      // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                            srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                            srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                        }

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                            srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                            srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                        }

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_mirror_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        Rpp32f srcPtrTemp_ps[24];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        Rpp32f srcPtrTemp_ps[24];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);    // cmn adjustment

                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                            Rpp32f srcPtrTemp_ps[8];
                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                            __m256 p[1];
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp16f) (((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                            srcPtrTemp -= vectorIncrementPerChannel;

                            Rpp32f srcPtrTemp_ps[8];
                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                            __m256 p[1];
                            rpp_simd_load(rpp_load8_f32_to_f32_mirror_avx, srcPtrTemp_ps, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);    // cmn adjustment

                            rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp16f) (((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        for(int pos = 0; pos < numRegs; pos += 2)
        {
            pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
            pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
            cmnParamLoc++;
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempR) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempG) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempB) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd store

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempR) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempG) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempB) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]] - 128);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                            rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTemp) + 128 * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]] - 128);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];
                            rpp_simd_load(rpp_load16_i8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTemp) + 128 * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]] - 128);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        // For PKD3-PKD3 case
        // set mean as multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]] 0 multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]]
        // set invStdDev as offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]] 1 offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]]
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            pCMNParams[0] = _mm256_setr_ps(multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f, multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f);
            pCMNParams[1] = _mm256_setr_ps(offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f, offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f);
        }
        else
        {
            for(int pos = 0; pos < numRegs; pos += 2)
            {
                pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
                pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
                cmnParamLoc++;
            }
        }

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel;
        Rpp32f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        *dstPtrTempG = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        *dstPtrTempB = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        *dstPtrTempG = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        *dstPtrTempB = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp32f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = ((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        dstPtrTemp[1] = ((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        dstPtrTemp[2] = ((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp32f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = ((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        dstPtrTemp[1] = ((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        dstPtrTemp[2] = ((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[8];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_rgb_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pkd3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        dstPtrTemp[1] = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        dstPtrTemp[2] = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[8];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_rgb_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pkd3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                        dstPtrTemp[1] = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                        dstPtrTemp[2] = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp32f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp32f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = ((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]];

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp32f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp32f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = ((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]];
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        // For PKD3-PKD3 case
        // set mean as multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]] 0 multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]]
        // set invStdDev as offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]] 1 offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]]
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            pCMNParams[0] = _mm256_setr_ps(multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f, multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f);
            pCMNParams[1] = _mm256_setr_ps(offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f, offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f);
        }
        else
        {
            for(int pos = 0; pos < numRegs; pos += 2)
            {
                pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
                pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
                cmnParamLoc++;
            }
        }

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel;
        Rpp16f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTemp += 3;
                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        *dstPtrTempG = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        *dstPtrTempB = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTempR++;
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRowR += dstDescPtr->strides.hStride;
                    dstPtrRowG += dstDescPtr->strides.hStride;
                    dstPtrRowB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp16f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp16f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[8];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_rgb_host(p, pCMNParams);   // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pkd3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[8];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_rgb_host(p, pCMNParams);    // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pkd3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp16f) (((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]]);
                        dstPtrTemp[1] = (Rpp16f) (((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]]);
                        dstPtrTemp[2] = (Rpp16f) (((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp16f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp16f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp16f) (((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp16f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp16f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp16f) (((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

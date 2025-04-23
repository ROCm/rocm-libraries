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

inline void compute_threshold_8_host(__m256 *p, __m256 *pThresholdParams)
{
    p[0] = rpp_pixel_check_0to1_avx(_mm256_blendv_ps(avx_p0, avx_p1, _mm256_and_ps(_mm256_cmp_ps(p[0], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[0], pThresholdParams[1],_CMP_LE_OQ))));
}
inline void compute_threshold_16_host(__m256 *p, __m256 *pThresholdParams)
{
    p[0] = _mm256_blendv_ps(avx_p0, avx_p255, _mm256_and_ps(_mm256_cmp_ps(p[0], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[0], pThresholdParams[1],_CMP_LE_OQ)));
    p[1] = _mm256_blendv_ps(avx_p0, avx_p255, _mm256_and_ps(_mm256_cmp_ps(p[1], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[1], pThresholdParams[1],_CMP_LE_OQ)));
}
inline void compute_threshold_24_host(__m256 *p, __m256 *pThresholdParams)
{
    __m256 pChannelCheck[3];
    pChannelCheck[0] = _mm256_and_ps(_mm256_cmp_ps(p[0], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[0], pThresholdParams[1],_CMP_LE_OQ));
    pChannelCheck[1] = _mm256_and_ps(_mm256_cmp_ps(p[1], pThresholdParams[2], _CMP_GE_OQ), _mm256_cmp_ps(p[1], pThresholdParams[3],_CMP_LE_OQ));
    pChannelCheck[2] = _mm256_and_ps(_mm256_cmp_ps(p[2], pThresholdParams[4], _CMP_GE_OQ), _mm256_cmp_ps(p[2], pThresholdParams[5],_CMP_LE_OQ));
    p[0] = rpp_pixel_check_0to1_avx(_mm256_blendv_ps(avx_p0, avx_p1, _mm256_and_ps(_mm256_and_ps(pChannelCheck[0], pChannelCheck[1]), pChannelCheck[2])));
    p[1] = p[0];
    p[2] = p[0];
}
inline void compute_threshold_48_host(__m256 *p, __m256 *pThresholdParams)
{
    __m256 pChannelCheck[3];
    pChannelCheck[0] = _mm256_and_ps(_mm256_cmp_ps(p[0], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[0], pThresholdParams[1],_CMP_LE_OQ));
    pChannelCheck[1] = _mm256_and_ps(_mm256_cmp_ps(p[2], pThresholdParams[2], _CMP_GE_OQ), _mm256_cmp_ps(p[2], pThresholdParams[3],_CMP_LE_OQ));
    pChannelCheck[2] = _mm256_and_ps(_mm256_cmp_ps(p[4], pThresholdParams[4], _CMP_GE_OQ), _mm256_cmp_ps(p[4], pThresholdParams[5],_CMP_LE_OQ));
    p[0] = _mm256_blendv_ps(avx_p0, avx_p255, _mm256_and_ps(_mm256_and_ps(pChannelCheck[0], pChannelCheck[1]), pChannelCheck[2]));
    p[2] = p[0];
    p[4] = p[0];
    pChannelCheck[0] = _mm256_and_ps(_mm256_cmp_ps(p[1], pThresholdParams[0], _CMP_GE_OQ), _mm256_cmp_ps(p[1], pThresholdParams[1],_CMP_LE_OQ));
    pChannelCheck[1] = _mm256_and_ps(_mm256_cmp_ps(p[3], pThresholdParams[2], _CMP_GE_OQ), _mm256_cmp_ps(p[3], pThresholdParams[3],_CMP_LE_OQ));
    pChannelCheck[2] = _mm256_and_ps(_mm256_cmp_ps(p[5], pThresholdParams[4], _CMP_GE_OQ), _mm256_cmp_ps(p[5], pThresholdParams[5],_CMP_LE_OQ));
    p[1] = _mm256_blendv_ps(avx_p0, avx_p255, _mm256_and_ps(_mm256_and_ps(pChannelCheck[0], pChannelCheck[1]), pChannelCheck[2]));
    p[3] = p[1];
    p[5] = p[1];
}

RppStatus threshold_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *minTensor,
                                      Rpp32f *maxTensor,
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

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        Rpp32u batchIndex = batchCount * srcDescPtr->c;
        Rpp32f minThreshold[3];
        Rpp32f maxThreshold[3];
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            minThreshold[c] = minTensor[batchIndex + c];
            maxThreshold[c] = maxTensor[batchIndex + c];
        }
#if __AVX2__
            __m256 pThresholdParams[6];
            for (int c = 0, i = 0; c < 3; c++, i += 2)
            {
                pThresholdParams[i] = _mm256_set1_ps(minThreshold[c]);
                pThresholdParams[i + 1] = _mm256_set1_ps(maxThreshold[c]);
            }
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                                          // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2]);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8u outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                                        // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8u outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);         // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                    // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2]);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8u outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 3 channel input
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + srcDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + srcDescPtr->strides.cStride;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);   // simd loads
                    compute_threshold_48_host(p, pThresholdParams);                                              // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores                                  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8u outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 1 channel input
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);      // simd loads
                    compute_threshold_16_host(p, pThresholdParams);              // threshold adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixel = static_cast<Rpp32f>(*srcPtrTemp++);
                    *dstPtrTemp++ = ((pixel >= minThreshold[0]) && (pixel <= maxThreshold[0])) ? 255 : 0;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus threshold_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *minTensor,
                                        Rpp32f *maxTensor,
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

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32u batchIndex = batchCount * srcDescPtr->c;
        Rpp32f minThreshold[3];
        Rpp32f maxThreshold[3];
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            minThreshold[c] = minTensor[batchIndex + c];
            maxThreshold[c] = maxTensor[batchIndex + c];
        }
#if __AVX2__
            __m256 pThresholdParams[6];
            for (int c = 0, i = 0; c < 3; c++, i += 2)
            {
                pThresholdParams[i] = _mm256_set1_ps(minThreshold[c]);
                pThresholdParams[i + 1] = _mm256_set1_ps(maxThreshold[c]);
            }
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                                           // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = srcPtrTemp[0];
                    pixelG = srcPtrTemp[1];
                    pixelB = srcPtrTemp[2];
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp32f outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                                         // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = *srcPtrTempR++;
                    pixelG = *srcPtrTempG++;
                    pixelB = *srcPtrTempB++;
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp32f outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);         // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                     // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = srcPtrTemp[0];
                    pixelG = srcPtrTemp[1];
                    pixelB = srcPtrTemp[2];
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp32f outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 3 channel input
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + srcDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + srcDescPtr->strides.cStride;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);   // simd loads
                    compute_threshold_24_host(p, pThresholdParams);                                               // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores                                  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = *srcPtrTempR++;
                    pixelG = *srcPtrTempG++;
                    pixelB = *srcPtrTempB++;
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp32f outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 1 channel input
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[1];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);      // simd loads
                    compute_threshold_8_host(p, pThresholdParams);              // threshold adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixel = *srcPtrTemp++;
                    *dstPtrTemp++ = ((pixel >= minThreshold[0]) && (pixel <= maxThreshold[0])) ? 1.0f : 0.0f;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus threshold_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *minTensor,
                                      Rpp32f *maxTensor,
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

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        Rpp32u batchIndex = batchCount * srcDescPtr->c;
        Rpp32f minThreshold[3];
        Rpp32f maxThreshold[3];
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            minThreshold[c] = minTensor[batchIndex + c] + static_cast<Rpp32f>(128);
            maxThreshold[c] = maxTensor[batchIndex + c] + static_cast<Rpp32f>(128);
        }
#if __AVX2__
            __m256 pThresholdParams[6];
            for (int c = 0, i = 0; c < 3; c++, i += 2)
            {
                pThresholdParams[i] = _mm256_set1_ps(minThreshold[c]);
                pThresholdParams[i + 1] = _mm256_set1_ps(maxThreshold[c]);
            }
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                                          // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0] + 128);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1] + 128);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2] + 128);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8s outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 127 : -128;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                                        // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++ + 128);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++ + 128);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++ + 128);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8s outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 127 : -128;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);         // simd loads
                    compute_threshold_48_host(p, pThresholdParams); 	                    // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0] + 128);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1] + 128);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2] + 128);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8s outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 127 : -128;
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 3 channel input
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + srcDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + srcDescPtr->strides.cStride;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);   // simd loads
                    compute_threshold_48_host(p, pThresholdParams);                                              // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores                                  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++ + 128);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++ + 128);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++ + 128);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp8s outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 127 : -128;
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 1 channel input
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);      // simd loads
                    compute_threshold_16_host(p, pThresholdParams);              // threshold adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixel = static_cast<Rpp32f>(*srcPtrTemp++ + 128);
                    *dstPtrTemp++ = ((pixel >= minThreshold[0]) && (pixel <= maxThreshold[0])) ? 127 : -128;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus threshold_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *minTensor,
                                        Rpp32f *maxTensor,
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

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32u batchIndex = batchCount * srcDescPtr->c;
        Rpp32f minThreshold[3];
        Rpp32f maxThreshold[3];
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            minThreshold[c] = minTensor[batchIndex + c];
            maxThreshold[c] = maxTensor[batchIndex + c];
        }
#if __AVX2__
            __m256 pThresholdParams[6];
            for (int c = 0, i = 0; c < 3; c++, i += 2)
            {
                pThresholdParams[i] = _mm256_set1_ps(minThreshold[c]);
                pThresholdParams[i + 1] = _mm256_set1_ps(maxThreshold[c]);
            }
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                                           // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2]);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp16f outVal = static_cast<Rpp16f>((channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f);
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                                         // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp16f outVal = static_cast<Rpp16f>((channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f);
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);         // simd loads
                    compute_threshold_24_host(p, pThresholdParams); 	                     // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2]);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp16f outVal = static_cast<Rpp16f>((channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f);
                    dstPtrTemp[0] = outVal;
                    dstPtrTemp[1] = outVal;
                    dstPtrTemp[2] = outVal;
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 3 channel input
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + srcDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + srcDescPtr->strides.cStride;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);   // simd loads
                    compute_threshold_24_host(p, pThresholdParams);                                               // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores                                  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB++);
                    channelCheck[0] = ((pixelR >= minThreshold[0]) &&  (pixelR <= maxThreshold[0]));
                    channelCheck[1] = ((pixelG >= minThreshold[1]) &&  (pixelG <= maxThreshold[1]));
                    channelCheck[2] = ((pixelB >= minThreshold[2]) &&  (pixelB <= maxThreshold[2]));
                    Rpp16f outVal = static_cast<Rpp16f>((channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 1.0f : 0.0f);
                    *dstPtrTempR++ = outVal;
                    *dstPtrTempG++ = outVal;
                    *dstPtrTempB++ = outVal;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold without fused output-layout toggle (NCHW -> NCHW) for 1 channel input
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[1];
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, p);      // simd loads
                    compute_threshold_8_host(p, pThresholdParams);              // threshold adjustment
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixel = static_cast<Rpp32f>(*srcPtrTemp++);
                    *dstPtrTemp++ = static_cast<Rpp16f>(((pixel >= minThreshold[0]) && (pixel <= maxThreshold[0])) ? 1.0f : 0.0f);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

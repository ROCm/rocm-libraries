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

inline void compute_brightness_48_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm256_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm256_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm256_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_48_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[6] = _mm_fmadd_ps(p[6], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[7] = _mm_fmadd_ps(p[7], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[8] = _mm_fmadd_ps(p[8], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[9] = _mm_fmadd_ps(p[9], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[10] = _mm_fmadd_ps(p[10], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[11] = _mm_fmadd_ps(p[11], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_24_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = rpp_pixel_check_0to1_avx(_mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]));    // brightness adjustment
    p[1] = rpp_pixel_check_0to1_avx(_mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]));    // brightness adjustment
    p[2] = rpp_pixel_check_0to1_avx(_mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]));    // brightness adjustment
}

inline void compute_brightness_24_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_16_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_16_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_12_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_8_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = rpp_pixel_check_0to1_avx(_mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]));    // brightness adjustment
}

inline void compute_brightness_8_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_4_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

RppStatus brightness_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
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
        Rpp32f beta = betaTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pBrightnessParams[2];
        pBrightnessParams[0] = _mm256_set1_ps(alpha);
        pBrightnessParams[1] = _mm256_set1_ps(beta);
#else
        __m128 pBrightnessParams[2];
        pBrightnessParams[0] = _mm_set1_ps(alpha);
        pBrightnessParams[1] = _mm_set1_ps(beta);
#endif

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[0])) * alpha) + beta));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[1])) * alpha) + beta));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (srcPtrTemp[2])) * alpha) + beta));

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempR)) * alpha) + beta));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempG)) * alpha) + beta));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTempB)) * alpha) + beta));

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;
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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_brightness_16_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                        compute_brightness_16_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*srcPtrTemp)) * alpha) + beta));

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
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
        Rpp32f beta = betaTensor[batchCount] * ONE_OVER_255;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pBrightnessParams[2];
        pBrightnessParams[0] = _mm256_set1_ps(alpha);
        pBrightnessParams[1] = _mm256_set1_ps(beta);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pBrightnessParams[2];
        pBrightnessParams[0] = _mm_set1_ps(alpha);
        pBrightnessParams[1] = _mm_set1_ps(beta);
#endif

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_brightness_24_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_brightness_12_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = RPPPIXELCHECKF32(srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = RPPPIXELCHECKF32(srcPtrTemp[2] * alpha + beta);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_24_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_12_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(*srcPtrTempB * alpha + beta);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

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
#if __AVX2__
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_brightness_8_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);    // simd loads
                        compute_brightness_4_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
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
        Rpp32f beta = betaTensor[batchCount] * ONE_OVER_255;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pBrightnessParams[2];
        pBrightnessParams[0] = _mm256_set1_ps(alpha);
        pBrightnessParams[1] = _mm256_set1_ps(beta);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pBrightnessParams[2];
        pBrightnessParams[0] = _mm_set1_ps(alpha);
        pBrightnessParams[1] = _mm_set1_ps(beta);
#endif

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_brightness_24_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_brightness_12_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#endif

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2] * alpha + beta);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
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
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_brightness_24_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_brightness_12_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB * alpha + beta);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

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
                        Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtrTemp_ps[cnt] = (Rpp16f) srcPtrTemp[cnt];
                        }
#if __AVX2__
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_brightness_8_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);    // simd loads
                        compute_brightness_4_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);    // simd stores
#endif

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                        }

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*srcPtrTemp) * alpha + beta);

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
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
        Rpp32f beta = betaTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pBrightnessParams[2];
        pBrightnessParams[0] = _mm256_set1_ps(alpha);
        pBrightnessParams[1] = _mm256_set1_ps(beta);
#else
        __m128 pBrightnessParams[2];
        pBrightnessParams[0] = _mm_set1_ps(alpha);
        pBrightnessParams[1] = _mm_set1_ps(beta);
#endif

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[0] + 128)) * alpha) + beta - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[1] + 128)) * alpha) + beta - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[2] + 128)) * alpha) + beta - 128);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_brightness_48_host(p, pBrightnessParams);  // brightness adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempR + 128)) * alpha) + beta - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempG + 128)) * alpha) + beta - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempB + 128)) * alpha) + beta - 128);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_brightness_16_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                        compute_brightness_16_host(p, pBrightnessParams);  // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores
#endif

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);

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
    }

    return RPP_SUCCESS;
}

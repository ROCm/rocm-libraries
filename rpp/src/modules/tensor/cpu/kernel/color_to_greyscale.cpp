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

inline void compute_color_48_to_greyscale_16_host(__m256 *p, __m256 *pChannelWeights)
{
    p[0] = _mm256_fmadd_ps(p[0], pChannelWeights[0], _mm256_fmadd_ps(p[2], pChannelWeights[1], _mm256_mul_ps(p[4], pChannelWeights[2])));
    p[1] = _mm256_fmadd_ps(p[1], pChannelWeights[0], _mm256_fmadd_ps(p[3], pChannelWeights[1], _mm256_mul_ps(p[5], pChannelWeights[2])));
}

inline void compute_color_48_to_greyscale_16_host(__m128 *p, __m128 *pChannelWeights)
{
    p[0] = _mm_fmadd_ps(p[0], pChannelWeights[0], _mm_fmadd_ps(p[4], pChannelWeights[1], _mm_mul_ps(p[8], pChannelWeights[2])));
    p[1] = _mm_fmadd_ps(p[1], pChannelWeights[0], _mm_fmadd_ps(p[5], pChannelWeights[1], _mm_mul_ps(p[9], pChannelWeights[2])));
    p[2] = _mm_fmadd_ps(p[2], pChannelWeights[0], _mm_fmadd_ps(p[6], pChannelWeights[1], _mm_mul_ps(p[10], pChannelWeights[2])));
    p[3] = _mm_fmadd_ps(p[3], pChannelWeights[0], _mm_fmadd_ps(p[7], pChannelWeights[1], _mm_mul_ps(p[11], pChannelWeights[2])));
}

inline void compute_color_24_to_greyscale_8_host(__m256 *p, __m256 *pChannelWeights)
{
    p[0] = _mm256_fmadd_ps(p[0], pChannelWeights[0], _mm256_fmadd_ps(p[1], pChannelWeights[1], _mm256_mul_ps(p[2], pChannelWeights[2])));
}

inline void compute_color_12_to_greyscale_4_host(__m128 *p, __m128 *pChannelWeights)
{
    p[0] = _mm_fmadd_ps(p[0], pChannelWeights[0], _mm_fmadd_ps(p[1], pChannelWeights[1], _mm_mul_ps(p[2], pChannelWeights[2])));
}

RppStatus color_to_greyscale_u8_u8_host_tensor(Rpp8u *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8u *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *channelWeights,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pChannelWeights[3];
        pChannelWeights[0] = _mm256_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm256_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm256_set1_ps(channelWeights[2]);
#else
        __m128 pChannelWeights[3];
        pChannelWeights[0] = _mm_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm_set1_ps(channelWeights[2]);
#endif

        // NHWC color to greyscale
        if (srcDescPtr->layout == RpptLayout::NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p); // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);       // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);       // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p); // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);   // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);       // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(std::fma((Rpp32f)srcPtrTemp[0], channelWeights[0], std::fma((Rpp32f)srcPtrTemp[1], channelWeights[1], (Rpp32f)srcPtrTemp[2] * channelWeights[2]))));
                    srcPtrTemp += 3;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // NCHW color to greyscale
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
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
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);                                   // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                   // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);                               // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(std::fma((Rpp32f)*srcPtrTempR, channelWeights[0], std::fma((Rpp32f)*srcPtrTempG, channelWeights[1], (Rpp32f)*srcPtrTempB * channelWeights[2]))));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_to_greyscale_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp32f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *channelWeights,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pChannelWeights[3];
        pChannelWeights[0] = _mm256_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm256_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm256_set1_ps(channelWeights[2]);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pChannelWeights[3];
        pChannelWeights[0] = _mm_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm_set1_ps(channelWeights[2]);
#endif

        // NHWC color to greyscale
        if (srcDescPtr->layout == RpptLayout::NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_color_24_to_greyscale_8_host(p, pChannelWeights);            // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);           // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_12_to_greyscale_4_host(p, pChannelWeights);        // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);           // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp = std::fma((Rpp32f)srcPtrTemp[0], channelWeights[0], std::fma((Rpp32f)srcPtrTemp[1], channelWeights[1], (Rpp32f)srcPtrTemp[2] * channelWeights[2]));
                    srcPtrTemp += 3;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // NCHW color to greyscale
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
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
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p); // simd loads
                    compute_color_24_to_greyscale_8_host(p, pChannelWeights);                                    // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);                                   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p); // simd loads
                    compute_color_12_to_greyscale_4_host(p, pChannelWeights);                                // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = std::fma((Rpp32f)*srcPtrTempR, channelWeights[0], std::fma((Rpp32f)*srcPtrTempG, channelWeights[1], (Rpp32f)*srcPtrTempB * channelWeights[2]));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_to_greyscale_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp16f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *channelWeights,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pChannelWeights[3];
        pChannelWeights[0] = _mm256_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm256_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm256_set1_ps(channelWeights[2]);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pChannelWeights[3];
        pChannelWeights[0] = _mm_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm_set1_ps(channelWeights[2]);
#endif

        // NHWC color to greyscale
        if (srcDescPtr->layout == RpptLayout::NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p); // simd loads
                    compute_color_24_to_greyscale_8_host(p, pChannelWeights);            // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);        // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_12_to_greyscale_4_host(p, pChannelWeights);        // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);           // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp = (Rpp16f) std::fma((Rpp32f)srcPtrTemp[0], channelWeights[0], std::fma((Rpp32f)srcPtrTemp[1], channelWeights[1], (Rpp32f)srcPtrTemp[2] * channelWeights[2]));
                    srcPtrTemp += 3;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // NCHW color to greyscale
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
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
                    compute_color_24_to_greyscale_8_host(p, pChannelWeights);                                                // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);                                            // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p); // simd loads
                    compute_color_12_to_greyscale_4_host(p, pChannelWeights);                                // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);                                   // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp16f) std::fma((Rpp32f)*srcPtrTempR, channelWeights[0], std::fma((Rpp32f)*srcPtrTempG, channelWeights[1], (Rpp32f)*srcPtrTempB * channelWeights[2]));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_to_greyscale_i8_i8_host_tensor(Rpp8s *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8s *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *channelWeights,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pChannelWeights[3];
        pChannelWeights[0] = _mm256_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm256_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm256_set1_ps(channelWeights[2]);
#else
        __m128 pChannelWeights[3];
        pChannelWeights[0] = _mm_set1_ps(channelWeights[0]);
        pChannelWeights[1] = _mm_set1_ps(channelWeights[1]);
        pChannelWeights[2] = _mm_set1_ps(channelWeights[2]);
#endif

        // NHWC color to greyscale
        if (srcDescPtr->layout == RpptLayout::NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p); // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);       // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);       // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p); // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);   // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);       // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(std::fma(((Rpp32f)srcPtrTemp[0] + 128), channelWeights[0], std::fma(((Rpp32f)srcPtrTemp[1] + 128), channelWeights[1], ((Rpp32f)srcPtrTemp[2] + 128) * channelWeights[2])) - 128);
                    srcPtrTemp += 3;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // NCHW color to greyscale
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
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
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);                                   // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);                                   // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_color_48_to_greyscale_16_host(p, pChannelWeights);                               // color_to_greyscale adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(std::fma(((Rpp32f)*srcPtrTempR + 128), channelWeights[0], std::fma(((Rpp32f)*srcPtrTempG + 128), channelWeights[1], ((Rpp32f)*srcPtrTempB + 128) * channelWeights[2])) - 128);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

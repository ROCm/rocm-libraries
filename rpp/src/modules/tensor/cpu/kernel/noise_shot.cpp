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
#include "rpp_cpu_random.hpp"
#include "rpp_cpu_simd_math.hpp"

inline Rpp32f rpp_host_math_exp_lim256approx(Rpp32f x)
{
  x = 1.0 + x * ONE_OVER_256;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;

  return x;
}

inline void compute_shot_noise_params_initialize_4_host_sse(Rpp32f &shotNoiseFactor, Rpp32f &shotNoiseFactorInv, __m128 &pShotNoiseFactor, __m128 &pShotNoiseFactorInv)
{
    pShotNoiseFactor = _mm_set1_ps(shotNoiseFactor);
    pShotNoiseFactorInv = _mm_set1_ps(shotNoiseFactorInv);
}

inline void compute_shot_noise_params_initialize_8_host_avx(Rpp32f &shotNoiseFactor, Rpp32f &shotNoiseFactorInv, __m256 &pShotNoiseFactor, __m256 &pShotNoiseFactorInv)
{
    pShotNoiseFactor = _mm256_set1_ps(shotNoiseFactor);
    pShotNoiseFactorInv = _mm256_set1_ps(shotNoiseFactorInv);
}

inline void compute_shot_noise_8_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    __m256 pShotNoiseValue = avx_p0;                                                                                                                                // Rpp32u shotNoiseValue = 0;
    __m256 pFactValue = fast_exp_avx(_mm256_mul_ps(*p, *pShotNoiseFactorInv));                                                                                      // Rpp32f factValue = expf(lambda);
    __m256 pFactValueMask = avx_p1;                                                                                                                                 // Set mask for pFactValue computation to 1
    do
    {
        pShotNoiseValue = _mm256_or_ps(_mm256_andnot_ps(pFactValueMask, pShotNoiseValue), _mm256_and_ps(pFactValueMask, _mm256_add_ps(pShotNoiseValue, avx_p1)));   // shotNoiseValue++;
        pFactValue = _mm256_mul_ps(pFactValue, rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter));                                                // factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);
        pFactValueMask = _mm256_cmp_ps(pFactValue, avx_p1, _CMP_GT_OQ);                                                                                             // compute new pFactValueMask for loop exit condition
    } while (_mm256_movemask_epi8(_mm256_cvtps_epi32(pFactValueMask)) != 0);                                                                                        // while (factValue > 1.0f);

    pShotNoiseValue = _mm256_sub_ps(pShotNoiseValue, avx_p1);                                                                                                       // shotNoiseValue -= 1;
    *p = _mm256_mul_ps(pShotNoiseValue, *pShotNoiseFactor);                                                                                                         // dst = pShotNoiseValue * shotNoiseFactor;
}

inline void compute_shot_noise_4_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    __m128 pShotNoiseValue = xmm_p0;                                                                                                                    // Rpp32u shotNoiseValue = 0;
    __m128 pFactValue = fast_exp_sse(_mm_mul_ps(*p, *pShotNoiseFactorInv));                                                                             // Rpp32f factValue = expf(lambda);
    __m128 pFactValueMask = xmm_p1;                                                                                                                     // Set mask for pFactValue computation to 1
    do
    {
        pShotNoiseValue = _mm_or_ps(_mm_andnot_ps(pFactValueMask, pShotNoiseValue), _mm_and_ps(pFactValueMask, _mm_add_ps(pShotNoiseValue, xmm_p1)));   // shotNoiseValue++;
        pFactValue = _mm_mul_ps(pFactValue, rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter));                                       // factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);
        pFactValueMask = _mm_cmpgt_ps(pFactValue, xmm_p1);                                                                                              // compute new pFactValueMask for loop exit condition
    } while (_mm_movemask_epi8(_mm_cvtps_epi32(pFactValueMask)) != 0);                                                                                  // while (factValue > 1.0f);

    pShotNoiseValue = _mm_sub_ps(pShotNoiseValue, xmm_p1);                                                                                              // shotNoiseValue -= 1;
    *p = _mm_mul_ps(pShotNoiseValue, *pShotNoiseFactor);                                                                                                // dst = pShotNoiseValue * shotNoiseFactor;
}

inline void compute_shot_noise_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  4, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  5, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  4, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  5, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  6, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  7, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  8, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  9, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p + 10, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p + 11, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline Rpp32u compute_shot_noise_1_host(RpptXorwowState *xorwowStatePtr, Rpp32f lambda)
{
    Rpp32u shotNoiseValue = 0;                                   // initialize shotNoiseValue to 0
    Rpp32f factValue = rpp_host_math_exp_lim256approx(lambda);   // initialize factValue to e^lambda
    do
    {
        shotNoiseValue++;                                        // additively cumulate shotNoiseValue by 1 until exit condition
        factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);    // multiplicatively cumulate factValue by the next uniform random number until exit condition
    } while (factValue > 1.0f);                                  // loop while factValue >= 1.0f

    return shotNoiseValue - 1;
}

RppStatus shot_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
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

        Rpp32f shotNoiseFactor = shotNoiseFactorTensor[batchCount];
        Rpp32f shotNoiseFactorInv = 1 / shotNoiseFactor;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

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

        RpptXorwowState xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_8_host_avx(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_4_host_sse(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#endif

        // Shot Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR = srcPtrTemp[0];
                        *dstPtrTempG = srcPtrTemp[1];
                        *dstPtrTempB = srcPtrTemp[2];
                    }

                    srcPtrTemp+=3;
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

        // Shot Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp8u src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};

                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) src[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) src[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) src[2] * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                        memcpy(dstPtrTemp, src, 3);

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

        // Shot Noise without fused output-layout toggle (NHWC -> NHWC)
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
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                        memcpy(dstPtrTemp, srcPtrTemp, 3);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) *srcPtrTempR * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) *srcPtrTempG * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) *srcPtrTempB * shotNoiseFactorInv;
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;

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
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, ((Rpp32f) *srcPtrTemp * shotNoiseFactorInv)) * shotNoiseFactor);
                    else
                        *dstPtrTemp = *srcPtrTemp;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus shot_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
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

        Rpp32f shotNoiseFactor = shotNoiseFactorTensor[batchCount] * ONE_OVER_255;
        Rpp32f shotNoiseFactorInv = 255.0f / shotNoiseFactorTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowState xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_8_host_avx(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_4_host_sse(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#endif

        // Shot Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        *dstPtrTempR = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR = srcPtrTemp[0];
                        *dstPtrTempG = srcPtrTemp[1];
                        *dstPtrTempB = srcPtrTemp[2];
                    }

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

        // Shot Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) *srcPtrTempR * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) *srcPtrTempG * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) *srcPtrTempB * shotNoiseFactorInv;
                        dstPtrTemp[0] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        dstPtrTemp[0] = *srcPtrTempR;
                        dstPtrTemp[1] = *srcPtrTempG;
                        dstPtrTemp[2] = *srcPtrTempB;
                    }

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

        // Shot Noise without fused output-layout toggle (NHWC -> NHWC)
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
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        dstPtrTemp[0] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        dstPtrTemp[0] = srcPtrTemp[0];
                        dstPtrTemp[1] = srcPtrTemp[1];
                        dstPtrTemp[2] = srcPtrTemp[2];
                    }

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) *srcPtrTempR++ * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) *srcPtrTempG++ * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) *srcPtrTempB++ * shotNoiseFactorInv;
                        *dstPtrTempR++ = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG++ = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB++ = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR++ = *srcPtrTempR++;
                        *dstPtrTempG++ = *srcPtrTempG++;
                        *dstPtrTempB++ = *srcPtrTempB++;
                    }
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif
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
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);    // simd stores
#else
                    __m128 p;
                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, &p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda;
                        poissonDistribLambda = (Rpp32f) *srcPtrTemp++ * shotNoiseFactorInv;
                        *dstPtrTemp++ = RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTemp++ = *srcPtrTemp++;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus shot_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
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

        Rpp32f shotNoiseFactor = shotNoiseFactorTensor[batchCount] * ONE_OVER_255;
        Rpp32f shotNoiseFactorInv = 255.0f / shotNoiseFactorTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowState xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_8_host_avx(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_4_host_sse(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#endif

        // Shot Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
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
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR = srcPtrTemp[0];
                        *dstPtrTempG = srcPtrTemp[1];
                        *dstPtrTempB = srcPtrTemp[2];
                    }

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

        // Shot Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
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
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) *srcPtrTempR * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) *srcPtrTempG * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) *srcPtrTempB * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        dstPtrTemp[0] = *srcPtrTempR;
                        dstPtrTemp[1] = *srcPtrTempG;
                        dstPtrTemp[2] = *srcPtrTempB;
                    }

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

        // Shot Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) srcPtrTemp[0] * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) srcPtrTemp[1] * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) srcPtrTemp[2] * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        dstPtrTemp[0] = srcPtrTemp[0];
                        dstPtrTemp[1] = srcPtrTemp[1];
                        dstPtrTemp[2] = srcPtrTemp[2];
                    }

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = (Rpp32f) *srcPtrTempR++ * shotNoiseFactorInv;
                        poissonDistribLambda[1] = (Rpp32f) *srcPtrTempG++ * shotNoiseFactorInv;
                        poissonDistribLambda[2] = (Rpp32f) *srcPtrTempB++ * shotNoiseFactorInv;
                        *dstPtrTempR++ = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor);
                        *dstPtrTempG++ = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor);
                        *dstPtrTempB++ = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTempR++ = *srcPtrTempR++;
                        *dstPtrTempG++ = *srcPtrTempG++;
                        *dstPtrTempB++ = *srcPtrTempB++;
                    }
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif
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
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, &p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, &p);    // simd stores
#else
                    __m128 p;
                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, &p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, &p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda;
                        poissonDistribLambda = (Rpp32f) *srcPtrTemp++ * shotNoiseFactorInv;
                        *dstPtrTemp++ = (Rpp16f) RPPPIXELCHECKF32(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda) * shotNoiseFactor);
                    }
                    else
                    {
                        *dstPtrTemp++ = *srcPtrTemp++;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus shot_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
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

        Rpp32f shotNoiseFactor = shotNoiseFactorTensor[batchCount];
        Rpp32f shotNoiseFactorInv = 1 / shotNoiseFactor;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

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

        RpptXorwowState xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_8_host_avx(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pShotNoiseFactor, pShotNoiseFactorInv;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_shot_noise_params_initialize_4_host_sse(shotNoiseFactor, shotNoiseFactorInv, pShotNoiseFactor, pShotNoiseFactorInv);
#endif

        // Shot Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = ((Rpp32f) srcPtrTemp[0] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[1] = ((Rpp32f) srcPtrTemp[1] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[2] = ((Rpp32f) srcPtrTemp[2] + 128.0f) * shotNoiseFactorInv;
                        *dstPtrTempR = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor) - 128);
                        *dstPtrTempG = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor) - 128);
                        *dstPtrTempB = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor) - 128);
                    }
                    else
                    {
                        *dstPtrTempR = srcPtrTemp[0];
                        *dstPtrTempG = srcPtrTemp[1];
                        *dstPtrTempB = srcPtrTemp[2];
                    }

                    srcPtrTemp+=3;
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

        // Shot Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp8s src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};

                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = ((Rpp32f) src[0] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[1] = ((Rpp32f) src[1] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[2] = ((Rpp32f) src[2] + 128.0f) * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor) - 128);
                        dstPtrTemp[1] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor) - 128);
                        dstPtrTemp[2] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor) - 128);
                    }
                    else
                        memcpy(dstPtrTemp, src, 3);

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

        // Shot Noise without fused output-layout toggle (NHWC -> NHWC)
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
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = ((Rpp32f) srcPtrTemp[0] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[1] = ((Rpp32f) srcPtrTemp[1] + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[2] = ((Rpp32f) srcPtrTemp[2] + 128.0f) * shotNoiseFactorInv;
                        dstPtrTemp[0] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor) - 128);
                        dstPtrTemp[1] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor) - 128);
                        dstPtrTemp[2] = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor) - 128);
                    }
                    else
                        memcpy(dstPtrTemp, srcPtrTemp, 3);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                    {
                        Rpp32f poissonDistribLambda[3];
                        poissonDistribLambda[0] = ((Rpp32f) *srcPtrTempR + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[1] = ((Rpp32f) *srcPtrTempG + 128.0f) * shotNoiseFactorInv;
                        poissonDistribLambda[2] = ((Rpp32f) *srcPtrTempB + 128.0f) * shotNoiseFactorInv;
                        *dstPtrTempR = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[0]) * shotNoiseFactor) - 128);
                        *dstPtrTempG = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[1]) * shotNoiseFactor) - 128);
                        *dstPtrTempB = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, poissonDistribLambda[2]) * shotNoiseFactor) - 128);
                    }
                    else
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Shot Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;

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
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                    if (shotNoiseFactor)
                        compute_shot_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, &pShotNoiseFactorInv, &pShotNoiseFactor);    // shot_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (shotNoiseFactor)
                        *dstPtrTemp = (Rpp8s) (RPPPIXELCHECK(compute_shot_noise_1_host(&xorwowState, (((Rpp32f) *srcPtrTemp + 128.0f) * shotNoiseFactorInv)) * shotNoiseFactor) - 128);
                    else
                        *dstPtrTemp = *srcPtrTemp;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

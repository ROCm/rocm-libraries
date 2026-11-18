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

inline void compute_salt_and_pepper_noise_params_initialize_4_host_sse(
    Rpp32f& noiseProbability, Rpp32f& saltProbability, Rpp32f salt, Rpp32f pepper,
    __m128* pSaltAndPepperNoiseParams) {
    pSaltAndPepperNoiseParams[0] = _mm_set1_ps(noiseProbability);
    pSaltAndPepperNoiseParams[1] = _mm_set1_ps(saltProbability);
    pSaltAndPepperNoiseParams[2] = _mm_set1_ps(salt);
    pSaltAndPepperNoiseParams[3] = _mm_set1_ps(pepper);
}

inline void compute_salt_and_pepper_noise_params_initialize_8_host_avx(
    Rpp32f& noiseProbability, Rpp32f& saltProbability, Rpp32f salt, Rpp32f pepper,
    __m256* pSaltAndPepperNoiseParams) {
    pSaltAndPepperNoiseParams[0] = _mm256_set1_ps(noiseProbability);
    pSaltAndPepperNoiseParams[1] = _mm256_set1_ps(saltProbability);
    pSaltAndPepperNoiseParams[2] = _mm256_set1_ps(salt);
    pSaltAndPepperNoiseParams[3] = _mm256_set1_ps(pepper);
}

inline void compute_salt_and_pepper_noise_8_host(__m256* p, __m256i* pxXorwowStateX,
                                                 __m256i* pxXorwowStateCounter,
                                                 __m256* pSaltAndPepperNoiseParams) {
    __m256 pMask[3];
    __m256 pRandomNumbers = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0], _CMP_GT_OQ);
    pMask[1] = _mm256_andnot_ps(
        pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_LE_OQ));
    pMask[2] = _mm256_andnot_ps(
        pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_GT_OQ));
    p[0] = _mm256_and_ps(pMask[0], p[0]);
    p[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], p[0]),
                        _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    p[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], p[0]),
                        _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_4_host(__m128* p, __m128i* pxXorwowStateX,
                                                 __m128i* pxXorwowStateCounter,
                                                 __m128* pSaltAndPepperNoiseParams) {
    __m128 pMask[3];
    __m128 pRandomNumbers = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0]);
    pMask[1] = _mm_andnot_ps(pMask[0], _mm_cmple_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pMask[2] = _mm_andnot_ps(pMask[0], _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    p[0] = _mm_and_ps(pMask[0], p[0]);
    p[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], p[0]),
                     _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    p[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], p[0]),
                     _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_16_host(__m256* p, __m256i* pxXorwowStateX,
                                                  __m256i* pxXorwowStateCounter,
                                                  __m256* pSaltAndPepperNoiseParams) {
    compute_salt_and_pepper_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_8_host(p + 1, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_16_host(__m128* p, __m128i* pxXorwowStateX,
                                                  __m128i* pxXorwowStateCounter,
                                                  __m128* pSaltAndPepperNoiseParams) {
    compute_salt_and_pepper_noise_4_host(p, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 1, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 2, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 3, pxXorwowStateX, pxXorwowStateCounter,
                                         pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_24_host(__m256* pR, __m256* pG, __m256* pB,
                                                  __m256i* pxXorwowStateX,
                                                  __m256i* pxXorwowStateCounter,
                                                  __m256* pSaltAndPepperNoiseParams) {
    __m256 pMask[3];
    __m256 pRandomNumbers = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0], _CMP_GT_OQ);
    pMask[1] = _mm256_andnot_ps(
        pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_LE_OQ));
    pMask[2] = _mm256_andnot_ps(
        pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_GT_OQ));
    pR[0] = _mm256_and_ps(pMask[0], pR[0]);
    pR[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pR[0]),
                         _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pR[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pR[0]),
                         _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pG[0] = _mm256_and_ps(pMask[0], pG[0]);
    pG[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pG[0]),
                         _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pG[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pG[0]),
                         _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pB[0] = _mm256_and_ps(pMask[0], pB[0]);
    pB[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pB[0]),
                         _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pB[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pB[0]),
                         _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_12_host(__m128* pR, __m128* pG, __m128* pB,
                                                  __m128i* pxXorwowStateX,
                                                  __m128i* pxXorwowStateCounter,
                                                  __m128* pSaltAndPepperNoiseParams) {
    __m128 pMask[3];
    __m128 pRandomNumbers = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0]);
    pMask[1] = _mm_andnot_ps(pMask[0], _mm_cmple_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pMask[2] = _mm_andnot_ps(pMask[0], _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pR[0] = _mm_and_ps(pMask[0], pR[0]);
    pR[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pR[0]),
                      _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pR[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pR[0]),
                      _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pG[0] = _mm_and_ps(pMask[0], pG[0]);
    pG[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pG[0]),
                      _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pG[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pG[0]),
                      _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pB[0] = _mm_and_ps(pMask[0], pB[0]);
    pB[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pB[0]),
                      _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pB[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pB[0]),
                      _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_48_host(__m256* p, __m256i* pxXorwowStateX,
                                                  __m256i* pxXorwowStateCounter,
                                                  __m256* pSaltAndPepperNoiseParams) {
    compute_salt_and_pepper_noise_24_host(p, p + 2, p + 4, pxXorwowStateX, pxXorwowStateCounter,
                                          pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_24_host(p + 1, p + 3, p + 5, pxXorwowStateX, pxXorwowStateCounter,
                                          pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_48_host(__m128* p, __m128i* pxXorwowStateX,
                                                  __m128i* pxXorwowStateCounter,
                                                  __m128* pSaltAndPepperNoiseParams) {
    compute_salt_and_pepper_noise_12_host(p, p + 4, p + 8, pxXorwowStateX, pxXorwowStateCounter,
                                          pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 1, p + 5, p + 9, pxXorwowStateX, pxXorwowStateCounter,
                                          pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 2, p + 6, p + 10, pxXorwowStateX,
                                          pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 3, p + 7, p + 11, pxXorwowStateX,
                                          pxXorwowStateCounter, pSaltAndPepperNoiseParams);
}

inline void salt_and_pepper_noise_u8_u8_host_impl(Rpp8u *srcPtrImage, RpptDescPtr srcDescPtr,
                                                   Rpp8u *dstPtrImage, RpptDescPtr dstDescPtr,
                                                   Rpp32f noiseProbability, Rpp32f saltProbability,
                                                   Rpp8u salt, Rpp8u pepper,
                                                   RpptXorwowState *xorwowInitialStatePtr,
                                                   Rpp32u offset, RpptROI roi,
                                                   RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp8u dst[3];
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dst, srcPtrTemp, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dst, salt, 3);
                else
                    memset(dst, pepper, 3);
                *dstPtrTempR = dst[0];
                *dstPtrTempG = dst[1];
                *dstPtrTempB = dst[2];

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp8u src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, src, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dstPtrTemp, salt, 3);
                else
                    memset(dstPtrTemp, pepper, 3);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8u *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, srcPtrTemp, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dstPtrTemp, salt, 3);
                else
                    memset(dstPtrTemp, pepper, 3);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability) {
                    *dstPtrTempR = *srcPtrTempR;
                    *dstPtrTempG = *srcPtrTempG;
                    *dstPtrTempB = *srcPtrTempB;
                } else {
                    if (randomNumberFloat <= saltProbability) {
                        *dstPtrTempR = salt;
                        *dstPtrTempG = salt;
                        *dstPtrTempB = salt;
                    } else {
                        *dstPtrTempR = pepper;
                        *dstPtrTempG = pepper;
                        *dstPtrTempB = pepper;
                    }
                }

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        alignedLength = bufferLength & ~15;

        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8u *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[2];
                rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);
                compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    *dstPtrTemp = *srcPtrTemp;
                else if (randomNumberFloat <= saltProbability)
                    *dstPtrTemp = salt;
                else
                    *dstPtrTemp = pepper;

                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
}

RppStatus salt_and_pepper_noise_u8_u8_host_tensor(
    Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr, RpptDescPtr dstDescPtr,
    Rpp32f* noiseProbabilityTensor, Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
    Rpp32f* pepperValueTensor, RpptXorwowState* xorwowInitialStatePtr, RpptROIPtr roiTensorPtrSrc,
    RpptRoiType roiType, RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp8u salt = (Rpp8u)(saltValueTensor[batchCount] * 255.0f);
        Rpp8u pepper = (Rpp8u)(pepperValueTensor[batchCount] * 255.0f);
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        salt_and_pepper_noise_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                               noiseProbability, saltProbability, salt, pepper,
                                               xorwowInitialStatePtr, offset, roi, layoutParams,
                                               intraThreads);
    }

    return RPP_SUCCESS;
}


inline void salt_and_pepper_noise_f32_f32_host_impl(Rpp32f *srcPtrImage, RpptDescPtr srcDescPtr,
                                                     Rpp32f *dstPtrImage, RpptDescPtr dstDescPtr,
                                                     Rpp32f noiseProbability, Rpp32f saltProbability,
                                                     Rpp32f salt, Rpp32f pepper,
                                                     RpptXorwowState *xorwowInitialStatePtr,
                                                     Rpp32u offset, RpptROI roi,
                                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

#if __AVX2__
    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;
#else
    Rpp32u alignedLength = (bufferLength / 12) * 12;
    Rpp32u vectorIncrement = 12;
    Rpp32u vectorIncrementPerChannel = 4;
#endif

    // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f dst[3];
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dst, srcPtrTemp, 12);
                else if (randomNumberFloat <= saltProbability)
                    std::fill_n(dst, 3, salt);
                else
                    std::fill_n(dst, 3, pepper);
                *dstPtrTempR = dst[0];
                *dstPtrTempG = dst[1];
                *dstPtrTempB = dst[2];

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, src, 12);
                else if (randomNumberFloat <= saltProbability)
                    std::fill_n(dstPtrTemp, 3, salt);
                else
                    std::fill_n(dstPtrTemp, 3, pepper);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, srcPtrTemp, 12);
                else if (randomNumberFloat <= saltProbability)
                    std::fill_n(dstPtrTemp, 3, salt);
                else
                    std::fill_n(dstPtrTemp, 3, pepper);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability) {
                    *dstPtrTempR = *srcPtrTempR;
                    *dstPtrTempG = *srcPtrTempG;
                    *dstPtrTempB = *srcPtrTempB;
                } else {
                    if (randomNumberFloat <= saltProbability) {
                        *dstPtrTempR = salt;
                        *dstPtrTempG = salt;
                        *dstPtrTempB = salt;
                    } else {
                        *dstPtrTempR = pepper;
                        *dstPtrTempG = pepper;
                        *dstPtrTempB = pepper;
                    }
                }

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
        alignedLength = bufferLength & ~7;
#else
        alignedLength = bufferLength & ~3;
#endif
        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p;
                rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);
                compute_salt_and_pepper_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter,
                                                     pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);
#else
                __m128 p;
                rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, &p);
                compute_salt_and_pepper_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter,
                                                     pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &p);
#endif
                srcPtrTemp += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    *dstPtrTemp = *srcPtrTemp;
                else if (randomNumberFloat <= saltProbability)
                    *dstPtrTemp = salt;
                else
                    *dstPtrTemp = pepper;

                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
}


// Save this output to a file and manually insert it into noise_salt_and_pepper.cpp
// Insert it right before 'RppStatus salt_and_pepper_noise_f32_f32_host_tensor('

// Generated F32 impl function: 414 lines
RppStatus salt_and_pepper_noise_f32_f32_host_tensor(
    Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
    Rpp32f* noiseProbabilityTensor, Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
    Rpp32f* pepperValueTensor, RpptXorwowState* xorwowInitialStatePtr, RpptROIPtr roiTensorPtrSrc,
    RpptRoiType roiType, RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp32f salt = saltValueTensor[batchCount];
        Rpp32f pepper = pepperValueTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        salt_and_pepper_noise_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                                 noiseProbability, saltProbability, salt, pepper,
                                                 xorwowInitialStatePtr, offset, roi, layoutParams,
                                                 intraThreads);
    }

    return RPP_SUCCESS;
}

inline void salt_and_pepper_noise_f16_f16_host_impl(Rpp16f *srcPtrImage, RpptDescPtr srcDescPtr,
                                                     Rpp16f *dstPtrImage, RpptDescPtr dstDescPtr,
                                                     Rpp32f noiseProbability, Rpp32f saltProbability,
                                                     Rpp16f salt, Rpp16f pepper,
                                                     RpptXorwowState *xorwowInitialStatePtr,
                                                     Rpp32u offset, RpptROI roi,
                                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

#if __AVX2__
    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;
#else
    Rpp32u alignedLength = (bufferLength / 12) * 12;
    Rpp32u vectorIncrement = 12;
    Rpp32u vectorIncrementPerChannel = 4;
#endif

    // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f16pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f16pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp16f dst[3];
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dst, srcPtrTemp, 6);
                else if (randomNumberFloat <= saltProbability) {
                    dst[0] = salt;
                    dst[1] = salt;
                    dst[2] = salt;
                } else {
                    dst[0] = pepper;
                    dst[1] = pepper;
                    dst[2] = pepper;
                }
                *dstPtrTempR = dst[0];
                *dstPtrTempG = dst[1];
                *dstPtrTempB = dst[2];

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f16pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f16pkd3, dstPtrTemp, p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp16f src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, src, 6);
                else if (randomNumberFloat <= saltProbability) {
                    dstPtrTemp[0] = salt;
                    dstPtrTemp[1] = salt;
                    dstPtrTemp[2] = salt;
                } else {
                    dstPtrTemp[0] = pepper;
                    dstPtrTemp[1] = pepper;
                    dstPtrTemp[2] = pepper;
                }

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp16f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f16pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f16pkd3, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, srcPtrTemp, 6);
                else if (randomNumberFloat <= saltProbability) {
                    dstPtrTemp[0] = salt;
                    dstPtrTemp[1] = salt;
                    dstPtrTemp[2] = salt;
                } else {
                    dstPtrTemp[0] = pepper;
                    dstPtrTemp[1] = pepper;
                    dstPtrTemp[2] = pepper;
                }

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load12_f16pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX,
                                                      &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store12_f32pln3_to_f16pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability) {
                    *dstPtrTempR = *srcPtrTempR;
                    *dstPtrTempG = *srcPtrTempG;
                    *dstPtrTempB = *srcPtrTempB;
                } else {
                    if (randomNumberFloat <= saltProbability) {
                        *dstPtrTempR = salt;
                        *dstPtrTempG = salt;
                        *dstPtrTempB = salt;
                    } else {
                        *dstPtrTempR = pepper;
                        *dstPtrTempG = pepper;
                        *dstPtrTempB = pepper;
                    }
                }

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
        alignedLength = bufferLength & ~7;
#else
        alignedLength = bufferLength & ~3;
#endif
        Rpp16f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp16f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p;
                rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, &p);
                compute_salt_and_pepper_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter,
                                                     pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p);
#else
                __m128 p;
                rpp_simd_load(rpp_load4_f16_to_f32, srcPtrTemp, &p);
                compute_salt_and_pepper_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter,
                                                     pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store4_f32_to_f16, dstPtrTemp, &p);
#endif
                srcPtrTemp += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    *dstPtrTemp = *srcPtrTemp;
                else if (randomNumberFloat <= saltProbability)
                    *dstPtrTemp = salt;
                else
                    *dstPtrTemp = pepper;

                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
}


RppStatus salt_and_pepper_noise_f16_f16_host_tensor(
    Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr, RpptDescPtr dstDescPtr,
    Rpp32f* noiseProbabilityTensor, Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
    Rpp32f* pepperValueTensor, RpptXorwowState* xorwowInitialStatePtr, RpptROIPtr roiTensorPtrSrc,
    RpptRoiType roiType, RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp16f salt = static_cast<Rpp16f>(saltValueTensor[batchCount]);
        Rpp16f pepper = static_cast<Rpp16f>(pepperValueTensor[batchCount]);
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        salt_and_pepper_noise_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                                 noiseProbability, saltProbability, salt, pepper,
                                                 xorwowInitialStatePtr, offset, roi, layoutParams,
                                                 intraThreads);
    }

    return RPP_SUCCESS;
}


inline void salt_and_pepper_noise_i8_i8_host_impl(Rpp8s *srcPtrImage, RpptDescPtr srcDescPtr,
                                                   Rpp8s *dstPtrImage, RpptDescPtr dstDescPtr,
                                                   Rpp32f noiseProbability, Rpp32f saltProbability,
                                                   Rpp8s salt, Rpp8s pepper,
                                                   RpptXorwowState *xorwowInitialStatePtr,
                                                   Rpp32u offset, RpptROI roi,
                                                   RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp8s dst[3];
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dst, srcPtrTemp, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dst, salt, 3);
                else
                    memset(dst, pepper, 3);
                *dstPtrTempR = dst[0];
                *dstPtrTempG = dst[1];
                *dstPtrTempB = dst[2];

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp8s src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, src, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dstPtrTemp, salt, 3);
                else
                    memset(dstPtrTemp, pepper, 3);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8s *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    memcpy(dstPtrTemp, srcPtrTemp, 3);
                else if (randomNumberFloat <= saltProbability)
                    memset(dstPtrTemp, salt, 3);
                else
                    memset(dstPtrTemp, pepper, 3);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
#else
                __m128 p[12];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);
                compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability) {
                    *dstPtrTempR = *srcPtrTempR;
                    *dstPtrTempG = *srcPtrTempG;
                    *dstPtrTempB = *srcPtrTempB;
                } else {
                    if (randomNumberFloat <= saltProbability) {
                        *dstPtrTempR = salt;
                        *dstPtrTempG = salt;
                        *dstPtrTempB = salt;
                    } else {
                        *dstPtrTempR = pepper;
                        *dstPtrTempG = pepper;
                        *dstPtrTempB = pepper;
                    }
                }

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        alignedLength = bufferLength & ~15;

        Rpp8s *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            RpptXorwowState xorwowState;
            Rpp32u rowOffset = offset + i * roi.xywhROI.roiWidth;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            __m256 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_8_host_avx(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            __m128 pSaltAndPepperNoiseParams[4];
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);
            compute_salt_and_pepper_noise_params_initialize_4_host_sse(
                noiseProbability, saltProbability, salt, pepper, pSaltAndPepperNoiseParams);
#endif

            Rpp8s *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[2];
                rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);
                compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);
#else
                __m128 p[4];
                rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);
                compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter,
                                                      pSaltAndPepperNoiseParams);
                rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);
#endif
                srcPtrTemp += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                if (randomNumberFloat > noiseProbability)
                    *dstPtrTemp = *srcPtrTemp;
                else if (randomNumberFloat <= saltProbability)
                    *dstPtrTemp = salt;
                else
                    *dstPtrTemp = pepper;

                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
}

RppStatus salt_and_pepper_noise_i8_i8_host_tensor(
    Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr, RpptDescPtr dstDescPtr,
    Rpp32f* noiseProbabilityTensor, Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
    Rpp32f* pepperValueTensor, RpptXorwowState* xorwowInitialStatePtr, RpptROIPtr roiTensorPtrSrc,
    RpptRoiType roiType, RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp8s salt = (Rpp8s)(saltValueTensor[batchCount] * 127.0f);
        Rpp8s pepper = (Rpp8s)(pepperValueTensor[batchCount] * 127.0f);
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        salt_and_pepper_noise_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                               noiseProbability, saltProbability, salt, pepper,
                                               xorwowInitialStatePtr, offset, roi, layoutParams,
                                               intraThreads);
    }

    return RPP_SUCCESS;
}

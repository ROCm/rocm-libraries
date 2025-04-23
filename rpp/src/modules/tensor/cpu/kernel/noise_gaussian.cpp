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

inline void compute_gaussian_noise_params_initialize_4_host_sse(Rpp32f &mean, Rpp32f &stdDev, __m128 *pGaussianNoiseParams)
{
    pGaussianNoiseParams[0] = _mm_set1_ps(mean);
    pGaussianNoiseParams[1] = _mm_set1_ps(stdDev);
}

inline void compute_gaussian_noise_params_initialize_8_host_avx(Rpp32f &mean, Rpp32f &stdDev, __m256 *pGaussianNoiseParams)
{
    pGaussianNoiseParams[0] = _mm256_set1_ps(mean);
    pGaussianNoiseParams[1] = _mm256_set1_ps(stdDev);
}

inline void compute_gaussian_noise_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    __m256 pRngVals[2], pSqrt[2];

    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm256_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pSqrt[0] = _mm256_sqrt_ps(p[0]);                                                                // pixSqrt = sqrt(pixVal);
    pSqrt[1] = _mm256_sqrt_ps(p[1]);                                                                // pixSqrt = sqrt(pixVal);
    p[0] = _mm256_fmadd_ps(pSqrt[0], pRngVals[0], p[0]);                                            // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
    p[1] = _mm256_fmadd_ps(pSqrt[1], pRngVals[1], p[1]);                                            // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    __m256 pRngVals[2], pSqrt[2];

    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm256_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    p[0] = _mm256_add_ps(p[0], pRngVals[0]);                                                        // return pixVal + rngVal;
    p[1] = _mm256_add_ps(p[1], pRngVals[1]);                                                        // return pixVal + rngVal;
}

inline void compute_gaussian_noise_8_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    __m128 pRngVals[2], pSqrt[2];

    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pSqrt[0] = _mm_sqrt_ps(p[0]);                                                               // pixSqrt = sqrt(pixVal);
    pSqrt[1] = _mm_sqrt_ps(p[1]);                                                               // pixSqrt = sqrt(pixVal);
    p[0] = _mm_fmadd_ps(pSqrt[0], pRngVals[0], p[0]);                                           // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
    p[1] = _mm_fmadd_ps(pSqrt[1], pRngVals[1], p[1]);                                           // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_8_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    __m128 pRngVals[2], pSqrt[2];

    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    p[0] = _mm_add_ps(p[0], pRngVals[0]);                                                       // return (pixVal + rngVal);
    p[1] = _mm_add_ps(p[1], pRngVals[1]);                                                       // return (pixVal + rngVal);
}

inline void compute_gaussian_noise_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_8_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_8_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[8], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[8], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m256 pRngVals[2], pSqrt;
    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pSqrt = _mm256_sqrt_ps(p[2]);                                                                   // pixSqrt = sqrt(pixVal);
    p[2] = _mm256_fmadd_ps(pSqrt, pRngVals[0], p[2]);                                               // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m256 pRngVals[2];
    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    p[2] = _mm256_add_ps(p[2], pRngVals[0]);                                                        // return pixVal + rngVal;
}

inline void compute_gaussian_noise_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m128 pRngVals[2], pSqrt;
    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pSqrt = _mm_sqrt_ps(p[2]);                                                                  // pixSqrt = sqrt(pixVal);
    p[2] = _mm_fmadd_ps(pSqrt, pRngVals[0], p[2]);                                              // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m128 pRngVals[2], pSqrt;
    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    p[2] = _mm_add_ps(p[2], pRngVals[0]);                                                       // return pixVal + rngVal;
}

inline Rpp32f compute_gaussian_noise_1_host(Rpp32f pixVal, RpptXorwowStateBoxMuller *xorwowStatePtr, Rpp32f mean, Rpp32f stdDev)
{
    Rpp32f rngVal, pixSqrt;
    rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    rngVal = rngVal * stdDev + mean;
    pixSqrt = sqrt(pixVal);

    return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline Rpp32f compute_gaussian_noise_voxel_1_host(Rpp32f pixVal, RpptXorwowStateBoxMuller *xorwowStatePtr, Rpp32f mean, Rpp32f stdDev)
{
    Rpp32f rngVal, pixSqrt;
    rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    rngVal = rngVal * stdDev + mean;
    return pixVal + rngVal;
}

RppStatus gaussian_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
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

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempG = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempB = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[1] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[2] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[1] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[2] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
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
                    *dstPtrTempR++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempG++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempB++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_multiply16_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                       // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);                                             // simd loads
                    rpp_multiply16_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = compute_gaussian_noise_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                    *dstPtrTempG = compute_gaussian_noise_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                    *dstPtrTempB = compute_gaussian_noise_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = compute_gaussian_noise_1_host(*srcPtrTempR, &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = compute_gaussian_noise_1_host(*srcPtrTempG, &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = compute_gaussian_noise_1_host(*srcPtrTempB, &xorwowState, mean, stdDev);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = compute_gaussian_noise_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = compute_gaussian_noise_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = compute_gaussian_noise_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
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
                    *dstPtrTempR++ = compute_gaussian_noise_1_host(*srcPtrTempR++, &xorwowState, mean, stdDev);
                    *dstPtrTempG++ = compute_gaussian_noise_1_host(*srcPtrTempG++, &xorwowState, mean, stdDev);
                    *dstPtrTempB++ = compute_gaussian_noise_1_host(*srcPtrTempB++, &xorwowState, mean, stdDev);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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

            Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                {
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);                                        // simd loads
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);                                      // simd stores
#else
                    __m128 p[2];
                    rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp, p);                                             // simd loads
                    compute_gaussian_noise_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannelDouble;
                    dstPtrTemp += vectorIncrementPerChannelDouble;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = compute_gaussian_noise_1_host(*srcPtrTemp++, &xorwowState, mean, stdDev);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
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
                    *dstPtrTempR = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0], &xorwowState, mean, stdDev);
                    *dstPtrTempG = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1], &xorwowState, mean, stdDev);
                    *dstPtrTempB = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2], &xorwowState, mean, stdDev);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);                                   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);                                   // simd stores
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
                    dstPtrTemp[0] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR, &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG, &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB, &xorwowState, mean, stdDev);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);                             // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);                           // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);                                 // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);                               // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0], &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1], &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2], &xorwowState, mean, stdDev);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
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
                    *dstPtrTempR++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++, &xorwowState, mean, stdDev);
                    *dstPtrTempG++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++, &xorwowState, mean, stdDev);
                    *dstPtrTempB++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++, &xorwowState, mean, stdDev);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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

            Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                {
                    Rpp32f srcPtrTemp_ps[16], dstPtrTemp_ps[16];
                    for(int cnt = 0; cnt < vectorIncrementPerChannelDouble; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp_ps, p);                                   // simd stores
#else
                    __m128 p[2];
                    rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp_ps, p);                                          // simd loads
                    compute_gaussian_noise_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp_ps, p);                                        // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannelDouble; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrementPerChannelDouble;
                    dstPtrTemp += vectorIncrementPerChannelDouble;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++, &xorwowState, mean, stdDev);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
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

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempG = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempB = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
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
                    *dstPtrTempR++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempG++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempB++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_multiply16_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);                                       // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);                                             // simd loads
                    rpp_multiply16_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptGenericDescPtr srcGenericDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptGenericDescPtr dstGenericDescPtr,
                                                 Rpp32f *meanTensor,
                                                 Rpp32f *stdDevTensor,
                                                 RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcGenericDescPtr->strides[0];
        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;
        bool copyInput = (!mean) && (!stdDev);
        if (copyInput)
        {
            copy_3d_host_tensor(srcPtrImage, srcGenericDescPtr, dstPtrImage, dstGenericDescPtr, &roi, layoutParams);
        }
        else
        {
            Rpp8u *srcPtrChannel, *dstPtrChannel;
            dstPtrChannel = dstPtrImage;

            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp32u vectorIncrementPerChannel = 16;
            RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
            __m256 pGaussianNoiseParams[2];
            pGaussianNoiseParams[0] = _mm256_set1_ps(mean);
            pGaussianNoiseParams[1] = _mm256_set1_ps(stdDev);
#else
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
            __m128 pGaussianNoiseParams[2];
            pGaussianNoiseParams[0] = _mm_set1_ps(mean);
            pGaussianNoiseParams[1] = _mm_set1_ps(stdDev);
#endif
            // Gaussian Noise without fused output-layout toggle (NDHWC -> NDHWC)
            if((srcGenericDescPtr->dims[4] == 3) && (srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
#if __AVX2__
                            __m256 p[6];
                            rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                            rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                            __m128 p[12];
                            rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                            rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                            srcPtrTemp += vectorIncrement;
                            dstPtrTemp += vectorIncrement;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                        {
                            dstPtrTemp[0] = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                            dstPtrTemp[1] = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                            dstPtrTemp[2] = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                            srcPtrTemp += 3;
                            dstPtrTemp += 3;
                        }
                        srcPtrRow += srcGenericDescPtr->strides[2];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[1];
                    dstPtrDepth += dstGenericDescPtr->strides[1];
                }
            }

            // Gaussian Noise without fused output-layout toggle (NCDHW -> NCDHW)
            else if ((srcGenericDescPtr->dims[1] == 3) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
                Rpp8u *srcPtrDepthR, *srcPtrDepthG, *srcPtrDepthB, *dstPtrDepthR, *dstPtrDepthG, *dstPtrDepthB;
                srcPtrDepthR = srcPtrChannel;
                srcPtrDepthG = srcPtrDepthR + srcGenericDescPtr->strides[1];
                srcPtrDepthB = srcPtrDepthG + srcGenericDescPtr->strides[1];
                dstPtrDepthR = dstPtrChannel;
                dstPtrDepthG = dstPtrDepthR + dstGenericDescPtr->strides[1];
                dstPtrDepthB = dstPtrDepthG + dstGenericDescPtr->strides[1];
                for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
                {
                    Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                    srcPtrRowR = srcPtrDepthR;
                    srcPtrRowG = srcPtrDepthG;
                    srcPtrRowB = srcPtrDepthB;
                    dstPtrRowR = dstPtrDepthR;
                    dstPtrRowG = dstPtrDepthG;
                    dstPtrRowB = dstPtrDepthB;
                    for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
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
                            rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                            rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                            __m128 p[12];
                            rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                            rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
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
                            *dstPtrTempR++ = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)*srcPtrTempR++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                            *dstPtrTempG++ = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)*srcPtrTempG++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                            *dstPtrTempB++ = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)*srcPtrTempB++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                        }
                        srcPtrRowR += srcGenericDescPtr->strides[3];
                        srcPtrRowG += srcGenericDescPtr->strides[3];
                        srcPtrRowB += srcGenericDescPtr->strides[3];
                        dstPtrRowR += srcGenericDescPtr->strides[3];
                        dstPtrRowG += srcGenericDescPtr->strides[3];
                        dstPtrRowB += srcGenericDescPtr->strides[3];
                    }
                    srcPtrDepthR += srcGenericDescPtr->strides[2];
                    srcPtrDepthG += srcGenericDescPtr->strides[2];
                    srcPtrDepthB += srcGenericDescPtr->strides[2];
                    dstPtrDepthR += srcGenericDescPtr->strides[2];
                    dstPtrDepthG += srcGenericDescPtr->strides[2];
                    dstPtrDepthB += srcGenericDescPtr->strides[2];
                }
            }

            // Gaussian Noise without fused output-layout toggle single channel (NCDHW -> NCDHW)
            else if ((srcGenericDescPtr->dims[1] == 1) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
                alignedLength = bufferLength & ~15;
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
#if __AVX2__
                            __m256 p[2];
                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                            rpp_multiply16_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply16_constant(p, avx_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                       // simd stores
#else
                            __m128 p[4];
                            rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);                                             // simd loads
                            rpp_multiply16_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                            compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_multiply16_constant(p, xmm_p255);                                                           // u8 un-normalization
                            rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);                                           // simd stores
#endif
                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (Rpp8u)(compute_gaussian_noise_voxel_1_host((Rpp32f)*srcPtrTemp++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                        }
                        srcPtrRow += srcGenericDescPtr->strides[3];
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[2];
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptGenericDescPtr srcGenericDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptGenericDescPtr dstGenericDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcGenericDescPtr->strides[0];
        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;
        bool copyInput = (!mean) && (!stdDev);
        if (copyInput)
        {
            copy_3d_host_tensor(srcPtrImage, srcGenericDescPtr, dstPtrImage, dstGenericDescPtr, &roi, layoutParams);
        }
        else
        {
            Rpp32f *srcPtrChannel, *dstPtrChannel;
            dstPtrChannel = dstPtrImage;
            RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32u vectorIncrement = 24;
            Rpp32u vectorIncrementPerChannel = 8;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);

            __m256 pGaussianNoiseParams[2];
            pGaussianNoiseParams[0] = _mm256_set1_ps(mean);
            pGaussianNoiseParams[1] = _mm256_set1_ps(stdDev);
#else
            Rpp32u alignedLength = (bufferLength / 12) * 12;
            Rpp32u vectorIncrement = 12;
            Rpp32u vectorIncrementPerChannel = 4;
            __m128i pxXorwowStateX[5], pxXorwowStateCounter;
            rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);

            __m128 pGaussianNoiseParams[2];
            pGaussianNoiseParams[0] = _mm_set1_ps(mean);
            pGaussianNoiseParams[1] = _mm_set1_ps(stdDev);
#endif
            // Gaussian Noise without fused output-layout toggle (NDHWC -> NDHWC)
            if((srcGenericDescPtr->dims[4] == 3) && (srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
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
#if __AVX2__
                            __m256 p[3];
                            rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                            compute_gaussian_noise_voxel_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                            __m128 p[4];
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                            compute_gaussian_noise_voxel_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                            srcPtrTemp += vectorIncrement;
                            dstPtrTemp += vectorIncrement;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                        {
                            dstPtrTemp[0] = compute_gaussian_noise_voxel_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                            dstPtrTemp[1] = compute_gaussian_noise_voxel_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                            dstPtrTemp[2] = compute_gaussian_noise_voxel_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);
                            srcPtrTemp += 3;
                            dstPtrTemp += 3;
                        }
                        srcPtrRow += srcGenericDescPtr->strides[2];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[1];
                    dstPtrDepth += dstGenericDescPtr->strides[1];
                }
            }

            // Gaussian Noise without fused output-layout toggle (NCDHW -> NCDHW)
            else if ((srcGenericDescPtr->dims[1] == 3) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
                Rpp32f *srcPtrDepthR, *srcPtrDepthG, *srcPtrDepthB, *dstPtrDepthR, *dstPtrDepthG, *dstPtrDepthB;
                srcPtrDepthR = srcPtrChannel;
                srcPtrDepthG = srcPtrDepthR + srcGenericDescPtr->strides[1];
                srcPtrDepthB = srcPtrDepthG + srcGenericDescPtr->strides[1];
                dstPtrDepthR = dstPtrChannel;
                dstPtrDepthG = dstPtrDepthR + dstGenericDescPtr->strides[1];
                dstPtrDepthB = dstPtrDepthG + dstGenericDescPtr->strides[1];
                for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
                {
                    Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                    srcPtrRowR = srcPtrDepthR;
                    srcPtrRowG = srcPtrDepthG;
                    srcPtrRowB = srcPtrDepthB;
                    dstPtrRowR = dstPtrDepthR;
                    dstPtrRowG = dstPtrDepthG;
                    dstPtrRowB = dstPtrDepthB;
                    for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
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
                            rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                            compute_gaussian_noise_voxel_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                            __m128 p[4];
                            rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                            compute_gaussian_noise_voxel_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
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
                            *dstPtrTempR++ = compute_gaussian_noise_voxel_1_host(*srcPtrTempR++, &xorwowState, mean, stdDev);
                            *dstPtrTempG++ = compute_gaussian_noise_voxel_1_host(*srcPtrTempG++, &xorwowState, mean, stdDev);
                            *dstPtrTempB++ = compute_gaussian_noise_voxel_1_host(*srcPtrTempB++, &xorwowState, mean, stdDev);
                        }
                        srcPtrRowR += srcGenericDescPtr->strides[3];
                        srcPtrRowG += srcGenericDescPtr->strides[3];
                        srcPtrRowB += srcGenericDescPtr->strides[3];
                        dstPtrRowR += srcGenericDescPtr->strides[3];
                        dstPtrRowG += srcGenericDescPtr->strides[3];
                        dstPtrRowB += srcGenericDescPtr->strides[3];
                    }
                    srcPtrDepthR += srcGenericDescPtr->strides[2];
                    srcPtrDepthG += srcGenericDescPtr->strides[2];
                    srcPtrDepthB += srcGenericDescPtr->strides[2];
                    dstPtrDepthR += srcGenericDescPtr->strides[2];
                    dstPtrDepthG += srcGenericDescPtr->strides[2];
                    dstPtrDepthB += srcGenericDescPtr->strides[2];
                }
            }

            // Gaussian Noise without fused output-layout toggle single channel (NCDHW -> NCDHW)
            else if ((srcGenericDescPtr->dims[1] == 1) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            {
                srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
#if __AVX2__
                alignedLength = bufferLength & ~15;
#else
                alignedLength = bufferLength & ~7;
#endif
                Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                        {
#if __AVX2__
                            __m256 p[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);                                        // simd loads
                            compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);                                      // simd stores
#else
                            __m128 p[2];
                            rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp, p);                                             // simd loads
                            compute_gaussian_noise_voxel_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp, p);                                           // simd stores
#endif
                            srcPtrTemp += vectorIncrementPerChannelDouble;
                            dstPtrTemp += vectorIncrementPerChannelDouble;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = compute_gaussian_noise_voxel_1_host(*srcPtrTemp++, &xorwowState, mean, stdDev);
                        }
                        srcPtrRow += srcGenericDescPtr->strides[3];
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[2];
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
            }
        }
    }

    return RPP_SUCCESS;
}

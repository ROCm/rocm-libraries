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

#include <random>
#include "spatter_mask.hpp"
#include "kernel_dims.hpp"
#include "host_tensor_executors.hpp"

inline void compute_spatter_48_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[3] = _mm256_fmadd_ps(p[3], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[4] = _mm256_fmadd_ps(p[4], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm256_fmadd_ps(p[5], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment
}

inline void compute_spatter_24_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_16_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment
}

inline void compute_spatter_8_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_48_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[0], pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[0], pSpatterMask[3]));    // spatter adjustment
    p[4] = _mm_fmadd_ps(p[4], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm_fmadd_ps(p[5], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[6] = _mm_fmadd_ps(p[6], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[1], pSpatterMask[2]));    // spatter adjustment
    p[7] = _mm_fmadd_ps(p[7], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[1], pSpatterMask[3]));    // spatter adjustment
    p[8] = _mm_fmadd_ps(p[8], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[9] = _mm_fmadd_ps(p[9], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment
    p[10] = _mm_fmadd_ps(p[10], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[2], pSpatterMask[2]));    // spatter adjustment
    p[11] = _mm_fmadd_ps(p[11], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[2], pSpatterMask[3]));    // spatter adjustment
}

inline void compute_spatter_16_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue, pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue, pSpatterMask[3]));    // spatter adjustment
}

inline void compute_spatter_12_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_4_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
}

RppStatus spatter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
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

        Rpp32f spatterValue[3];
        spatterValue[0] = (Rpp32f) spatterColor.B;
        spatterValue[1] = (Rpp32f) spatterColor.G;
        spatterValue[2] = (Rpp32f) spatterColor.R;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, SPATTER_MAX_WIDTH - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, SPATTER_MAX_HEIGHT - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *spatterMaskPtr, *spatterMaskInvPtr;
        spatterMaskPtr = &spatterMask[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];
        spatterMaskInvPtr = &spatterMaskInv[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        if (srcDescPtr->c == 1)
            spatterValue[0] = spatterValue[1] = spatterValue[2] = (spatterValue[0] + spatterValue[1] + spatterValue[2]) * 0.3333;

#if __AVX2__
        __m256 pSpatterValue[3];
        pSpatterValue[0] = _mm256_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm256_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm256_set1_ps(spatterValue[2]);
#else
        __m128 pSpatterValue[3];
        pSpatterValue[0] = _mm_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm_set1_ps(spatterValue[2]);
#endif

        // Spatter without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[0]) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[1]) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[2]) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempR) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempG) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempB) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTemp) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15) - 16;

            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
#endif
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
#if __AVX2__
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrChannel, p);    // simd loads
                        compute_spatter_16_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrChannel, p);    // simd stores
#else
                        __m128 p[4];
                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrChannel, p);    // simd loads
                        compute_spatter_16_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrChannel, p);    // simd stores
#endif
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrChannel) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp));
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus spatter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
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

        Rpp32f spatterValue[3];
        spatterValue[0] = (Rpp32f) spatterColor.B * ONE_OVER_255;
        spatterValue[1] = (Rpp32f) spatterColor.G * ONE_OVER_255;
        spatterValue[2] = (Rpp32f) spatterColor.R * ONE_OVER_255;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, SPATTER_MAX_WIDTH - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, SPATTER_MAX_HEIGHT - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *spatterMaskPtr, *spatterMaskInvPtr;
        spatterMaskPtr = &spatterMask[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];
        spatterMaskInvPtr = &spatterMaskInv[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];

        if (srcDescPtr->c == 1)
            spatterValue[0] = spatterValue[1] = spatterValue[2] = (spatterValue[0] + spatterValue[1] + spatterValue[2]) * 0.3333;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pSpatterValue[3];
        pSpatterValue[0] = _mm256_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm256_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm256_set1_ps(spatterValue[2]);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pSpatterValue[3];
        pSpatterValue[0] = _mm_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm_set1_ps(spatterValue[2]);
#endif

        // Spatter without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32((srcPtrTemp[0]) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp);
                    *dstPtrTempG = RPPPIXELCHECKF32((srcPtrTemp[1]) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp);
                    *dstPtrTempB = RPPPIXELCHECKF32((srcPtrTemp[2]) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[4];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[4];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif

            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
#else
                    __m128 pSpatterMask, pSpatterMaskInv;
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
#endif
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
#if __AVX2__
                        __m256 p;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrChannel, &p);    // simd loads
                        compute_spatter_8_host(&p, &pSpatterMaskInv, &pSpatterMask, &pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrChannel, &p);    // simd stores
#else
                        __m128 p;
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrChannel, &p);    // simd loads
                        compute_spatter_4_host(&p, &pSpatterMaskInv, &pSpatterMask, &pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrChannel, &p);    // simd stores
#endif
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = RPPPIXELCHECKF32((*srcPtrChannel) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp);
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus spatter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
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

        Rpp32f spatterValue[3];
        spatterValue[0] = (Rpp32f) spatterColor.B * ONE_OVER_255;
        spatterValue[1] = (Rpp32f) spatterColor.G * ONE_OVER_255;
        spatterValue[2] = (Rpp32f) spatterColor.R * ONE_OVER_255;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, SPATTER_MAX_WIDTH - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, SPATTER_MAX_HEIGHT - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *spatterMaskPtr, *spatterMaskInvPtr;
        spatterMaskPtr = &spatterMask[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];
        spatterMaskInvPtr = &spatterMaskInv[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];

        if (srcDescPtr->c == 1)
            spatterValue[0] = spatterValue[1] = spatterValue[2] = (spatterValue[0] + spatterValue[1] + spatterValue[2]) * 0.3333;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pSpatterValue[3];
        pSpatterValue[0] = _mm256_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm256_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm256_set1_ps(spatterValue[2]);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pSpatterValue[3];
        pSpatterValue[0] = _mm_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm_set1_ps(spatterValue[2]);
#endif

        // Spatter without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
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
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[0]) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[1]) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[2]) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

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
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[4];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempR) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempG) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempB) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_spatter_24_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 pSpatterMask, pSpatterMaskInv, p[4];
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_spatter_12_host(p, &pSpatterMaskInv, &pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTemp) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif

            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask, pSpatterMaskInv;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
#else
                    __m128 pSpatterMask, pSpatterMaskInv;
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskPtrTemp, &pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load4_f32_to_f32, spatterMaskInvPtrTemp, &pSpatterMaskInv);    // simd loads
#endif
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        Rpp32f srcPtrChannel_ps[8], dstPtrChannel_ps[8];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            srcPtrChannel_ps[cnt] = (Rpp32f) srcPtrChannel[cnt];
#if __AVX2__
                        __m256 p;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrChannel_ps, &p);    // simd loads
                        compute_spatter_8_host(&p, &pSpatterMaskInv, &pSpatterMask, &pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrChannel_ps, &p);    // simd stores
#else
                        __m128 p;
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrChannel_ps, &p);    // simd loads
                        compute_spatter_4_host(&p, &pSpatterMaskInv, &pSpatterMask, &pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrChannel_ps, &p);    // simd stores
#endif
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            dstPtrChannel[cnt] = (Rpp16f) dstPtrChannel_ps[cnt];
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrChannel) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp);
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus spatter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
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

        Rpp32f spatterValue[3];
        spatterValue[0] = (Rpp32f) spatterColor.B;
        spatterValue[1] = (Rpp32f) spatterColor.G;
        spatterValue[2] = (Rpp32f) spatterColor.R;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, SPATTER_MAX_WIDTH - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, SPATTER_MAX_HEIGHT - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *spatterMaskPtr, *spatterMaskInvPtr;
        spatterMaskPtr = &spatterMask[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];
        spatterMaskInvPtr = &spatterMaskInv[(SPATTER_MAX_WIDTH * maskLoc.y) + maskLoc.x];

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        if (srcDescPtr->c == 1)
            spatterValue[0] = spatterValue[1] = spatterValue[2] = (spatterValue[0] + spatterValue[1] + spatterValue[2]) * 0.3333;

#if __AVX2__
        __m256 pSpatterValue[3];
        pSpatterValue[0] = _mm256_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm256_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm256_set1_ps(spatterValue[2]);
#else
        __m128 pSpatterValue[3];
        pSpatterValue[0] = _mm_set1_ps(spatterValue[0]);
        pSpatterValue[1] = _mm_set1_ps(spatterValue[1]);
        pSpatterValue[2] = _mm_set1_ps(spatterValue[2]);
#endif

        // Spatter without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[0] + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp) - 128.0f);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[1] + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp) - 128.0f);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[2] + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp) - 128.0f);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempR + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[0] * *spatterMaskPtrTemp) - 128.0f);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempG + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[1] * *spatterMaskPtrTemp) - 128.0f);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempB + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[2] * *spatterMaskPtrTemp) - 128.0f);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4], p[12];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_spatter_48_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue);    // spatter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTemp + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp) - 128.0f);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }

        // Spatter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15) - 16;

            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *spatterMaskPtrRow, *spatterMaskInvPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            spatterMaskPtrRow = spatterMaskPtr;
            spatterMaskInvPtrRow = spatterMaskInvPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *spatterMaskPtrTemp, *spatterMaskInvPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                spatterMaskPtrTemp = spatterMaskPtrRow;
                spatterMaskInvPtrTemp = spatterMaskInvPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pSpatterMask[2], pSpatterMaskInv[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
#else
                    __m128 pSpatterMask[4], pSpatterMaskInv[4];
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskPtrTemp, pSpatterMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32, spatterMaskInvPtrTemp, pSpatterMaskInv);    // simd loads
#endif
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
#if __AVX2__
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrChannel, p);    // simd loads
                        compute_spatter_16_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrChannel, p);    // simd stores
#else
                        __m128 p[4];
                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrChannel, p);    // simd loads
                        compute_spatter_16_host(p, pSpatterMaskInv, pSpatterMask, pSpatterValue[c]);    // spatter adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrChannel, p);    // simd stores
#endif
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    spatterMaskPtrTemp += vectorIncrementPerChannel;
                    spatterMaskInvPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrChannel + 128.0f) * *spatterMaskInvPtrTemp + spatterValue[c] * *spatterMaskPtrTemp) - 128.0f);
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    spatterMaskPtrTemp++;
                    spatterMaskInvPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                spatterMaskPtrRow += SPATTER_MAX_WIDTH;
                spatterMaskInvPtrRow += SPATTER_MAX_WIDTH;
            }
        }
    }

    return RPP_SUCCESS;
}

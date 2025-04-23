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
#include <random>
#include "rpp_cpu_simd_math.hpp"

inline void compute_fog_48_host(__m256 *p, __m256 *pFogAlphaMask, __m256 *pFogIntensityMask, __m256 pIntensityFactor, __m256 pGrayFactor, __m256 *pConversionFactor)
{
    __m256 pAlphaMaskFactor[2], pIntensityMaskFactor[2], pGray[2], pOneMinusGrayFactor;
    pGray[0] = _mm256_fmadd_ps(pConversionFactor[0], p[0], _mm256_fmadd_ps(pConversionFactor[1], p[2], _mm256_mul_ps(pConversionFactor[2], p[4])));
    pGray[1] = _mm256_fmadd_ps(pConversionFactor[0], p[1], _mm256_fmadd_ps(pConversionFactor[1], p[3], _mm256_mul_ps(pConversionFactor[2], p[5])));
    pOneMinusGrayFactor = _mm256_sub_ps(avx_p1 , pGrayFactor);
    pGray[0] = _mm256_mul_ps(pGray[0], pGrayFactor);
    pGray[1] = _mm256_mul_ps(pGray[1], pGrayFactor);
    p[0] = _mm256_fmadd_ps(p[0], pOneMinusGrayFactor, pGray[0]);
    p[1] = _mm256_fmadd_ps(p[1], pOneMinusGrayFactor, pGray[0]);
    p[2] = _mm256_fmadd_ps(p[2], pOneMinusGrayFactor, pGray[0]);
    p[3] = _mm256_fmadd_ps(p[3], pOneMinusGrayFactor, pGray[1]);
    p[4] = _mm256_fmadd_ps(p[4], pOneMinusGrayFactor, pGray[1]);
    p[5] = _mm256_fmadd_ps(p[5], pOneMinusGrayFactor, pGray[1]);
    pAlphaMaskFactor[0] = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pAlphaMaskFactor[1] = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[1], pIntensityFactor));
    pIntensityMaskFactor[0] = _mm256_mul_ps(pFogIntensityMask[0], _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pIntensityMaskFactor[1] = _mm256_mul_ps(pFogIntensityMask[1], _mm256_add_ps(pFogAlphaMask[1], pIntensityFactor));
    p[0] = _mm256_fmadd_ps(p[0],  pAlphaMaskFactor[0], pIntensityMaskFactor[0]);    // fog adjustment
    p[1] = _mm256_fmadd_ps(p[1],  pAlphaMaskFactor[1], pIntensityMaskFactor[1]);    // fog adjustment
    p[2] = _mm256_fmadd_ps(p[2],  pAlphaMaskFactor[0], pIntensityMaskFactor[0]);    // fog adjustment
    p[3] = _mm256_fmadd_ps(p[3],  pAlphaMaskFactor[1], pIntensityMaskFactor[1]);    // fog adjustment
    p[4] = _mm256_fmadd_ps(p[4],  pAlphaMaskFactor[0], pIntensityMaskFactor[0]);    // fog adjustment
    p[5] = _mm256_fmadd_ps(p[5],  pAlphaMaskFactor[1], pIntensityMaskFactor[1]);    // fog adjustment
}

inline void compute_fog_24_host(__m256 *p, __m256 *pFogAlphaMask, __m256 *pFogIntensityMask, __m256 pIntensityFactor, __m256 pGrayFactor, __m256 *pConversionFactor)
{
    __m256 pAlphaMaskFactor, pIntensityMaskFactor, pGray, pOneMinusGrayFactor;
    pGray = _mm256_fmadd_ps(pConversionFactor[0], p[0], _mm256_fmadd_ps(pConversionFactor[1], p[1], _mm256_mul_ps(pConversionFactor[2], p[2])));
    pOneMinusGrayFactor = _mm256_sub_ps(avx_p1 , pGrayFactor);
    pGray = _mm256_mul_ps(pGray, pGrayFactor);
    p[0] = _mm256_fmadd_ps(p[0], pOneMinusGrayFactor, pGray);
    p[1] = _mm256_fmadd_ps(p[1], pOneMinusGrayFactor, pGray);
    p[2] = _mm256_fmadd_ps(p[2], pOneMinusGrayFactor, pGray);
    pAlphaMaskFactor = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pIntensityMaskFactor = _mm256_mul_ps(pFogIntensityMask[0], _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    p[0] = _mm256_fmadd_ps(p[0],  pAlphaMaskFactor, pIntensityMaskFactor);    // fog adjustment
    p[1] = _mm256_fmadd_ps(p[1],  pAlphaMaskFactor, pIntensityMaskFactor);    // fog adjustment
    p[2] = _mm256_fmadd_ps(p[2],  pAlphaMaskFactor, pIntensityMaskFactor);    // fog adjustment
}

inline void compute_fog_16_host(__m256 *p, __m256 *pFogAlphaMask, __m256 *pFogIntensityMask, __m256 pIntensityFactor)
{
    __m256 pAlphaMaskFactor[2], pIntensityMaskFactor[2];
    pAlphaMaskFactor[0] = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pAlphaMaskFactor[1] = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[1], pIntensityFactor));
    pIntensityMaskFactor[0] = _mm256_mul_ps(pFogIntensityMask[0], _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pIntensityMaskFactor[1] = _mm256_mul_ps(pFogIntensityMask[1], _mm256_add_ps(pFogAlphaMask[1], pIntensityFactor));
    p[0] = _mm256_fmadd_ps(p[0],  pAlphaMaskFactor[0], pIntensityMaskFactor[0]);    // fog adjustment
    p[1] = _mm256_fmadd_ps(p[1],  pAlphaMaskFactor[1], pIntensityMaskFactor[1]);    // fog adjustment
}

inline void compute_fog_8_host(__m256 *p, __m256 *pFogAlphaMask, __m256 *pFogIntensityMask, __m256 pIntensityFactor)
{
    __m256 pAlphaMaskFactor, pIntensityMaskFactor;
    pAlphaMaskFactor = _mm256_sub_ps(avx_p1, _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    pIntensityMaskFactor = _mm256_mul_ps(pFogIntensityMask[0], _mm256_add_ps(pFogAlphaMask[0], pIntensityFactor));
    p[0] = _mm256_fmadd_ps(p[0],  pAlphaMaskFactor, pIntensityMaskFactor);    // fog adjustment
}

RppStatus fog_u8_u8_host_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *fogAlphaMask,
                                Rpp32f *fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensityValue = intensityFactor[batchCount];
        Rpp32f grayValue = grayFactor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
#if __AVX2__
        __m256 pIntensity = _mm256_set1_ps(intensityValue);
        __m256 pGrayFactor = _mm256_set1_ps(grayValue);
        __m256 pConversionFactor[3];
        pConversionFactor[0] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_RED);
        pConversionFactor[1] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_GREEN);
        pConversionFactor[2] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_BLUE);
#endif
        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                         // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);            // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]),
                                          static_cast<Rpp32f>(srcPtrTemp[1]),
                                          static_cast<Rpp32f>(srcPtrTemp[2])};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.R * alphaMaskFactor) + intensityMaskFactor)));
                    *dstPtrTempG++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.G * alphaMaskFactor) + intensityMaskFactor)));
                    *dstPtrTempB++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.B * alphaMaskFactor) + intensityMaskFactor)));
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);              // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR),
                                          static_cast<Rpp32f>(*srcPtrTempG),
                                          static_cast<Rpp32f>(*srcPtrTempB)};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.R * alphaMaskFactor) + intensityMaskFactor)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.G * alphaMaskFactor) + intensityMaskFactor)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.B * alphaMaskFactor) + intensityMaskFactor)));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                         // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]),
                                          static_cast<Rpp32f>(srcPtrTemp[1]),
                                          static_cast<Rpp32f>(srcPtrTemp[2])};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.R * alphaMaskFactor) + intensityMaskFactor)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.G * alphaMaskFactor) + intensityMaskFactor)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf((pixel.B * alphaMaskFactor) + intensityMaskFactor)));
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);              // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads                                                        // simd normalize
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);            // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;

                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR),
                                          static_cast<Rpp32f>(*srcPtrTempG),
                                          static_cast<Rpp32f>(*srcPtrTempB)};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (*fogIntensityMaskPtrTemp++ * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((pixel.R * alphaMaskFactor) + intensityMaskFactor)));
                    *dstPtrTempG++ = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((pixel.G * alphaMaskFactor) + intensityMaskFactor)));
                    *dstPtrTempB++ = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((pixel.B * alphaMaskFactor) + intensityMaskFactor)));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15);

            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2],p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                   // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);           // simd loads
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity);                           // fog adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (*fogIntensityMaskPtrTemp++ * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrTemp) * alphaMaskFactor + intensityMaskFactor)));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f16_f16_host_tensor(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensityValue = intensityFactor[batchCount];
        Rpp32f grayValue = grayFactor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#if __AVX2__
        __m256 pIntensity = _mm256_set1_ps(intensityValue);
        __m256 pGrayFactor = _mm256_set1_ps(grayValue);
        __m256 pConversionFactor[3];
        pConversionFactor[0] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_RED);
        pConversionFactor[1] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_GREEN);
        pConversionFactor[2] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_BLUE);
#endif
        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                                        // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0.1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);           // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);             // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0.1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                                      // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempG) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempB) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                                        // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0.1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                                      // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);             // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalize to range [0,255]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // snow adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalize to range [0,1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);           // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempG) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempB) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#endif

            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                   // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);           // simd loads
                    pFogIntensityMask = _mm256_mul_ps(pFogIntensityMask, avx_p1op255);                              // u8 normalization to range[0,1];
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, &p);                                        // simd loads
                    compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask, pIntensity);                         // fog adjustment
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p);                                      // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrTemp) * alphaMaskFactor + intensityMaskFactor));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f32_f32_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensityValue = intensityFactor[batchCount];
        Rpp32f grayValue = grayFactor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;                                              // Random number engine seed
        std::mt19937 gen(rd());                                             // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#if __AVX2__
        __m256 pIntensity = _mm256_set1_ps(intensityValue);
        __m256 pGrayFactor = _mm256_set1_ps(grayValue);
        __m256 pConversionFactor[3];
        pConversionFactor[0] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_RED);
        pConversionFactor[1] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_GREEN);
        pConversionFactor[2] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_BLUE);
#endif
        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                        // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0.1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);           // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);             // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0,1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                                      // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempG) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempB) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                        // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalization
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalization to range[0.1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                                      // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) * 255,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);             // simd loads
                    rpp_multiply24_constant(p, avx_p255);                                                                   // denormalize range from [0,255]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor); // fog adjustment
                    rpp_normalize24_avx(p);                                                                                 // normalize range from [0 ,1]
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);           // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempG) * 255,
                                          static_cast<Rpp32f>(*srcPtrTempB) * 255};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = ((pixel.R * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.G = ((pixel.G * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    pixel.B = ((pixel.B * oneMinusGrayValue) + gray) * ONE_OVER_255;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32((pixel.B * alphaMaskFactor) + intensityMaskFactor));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {

            alignedLength = bufferLength & ~7;

            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);                   // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);           // simd loads
                    pFogIntensityMask = _mm256_mul_ps(pFogIntensityMask, avx_p1op255);                              // u8 normalization to range[0,1]
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);                                        // simd loads
                    compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask, pIntensity);                         // fog adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);                                      // simd stores

                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = (((*fogIntensityMaskPtrTemp++) * ONE_OVER_255) * ((*fogAlphaMaskPtrTemp++) + intensityValue));
                    *dstPtrTemp++ = RPPPIXELCHECKF32((*srcPtrTemp) * alphaMaskFactor + intensityMaskFactor);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f* fogAlphaMask,
                                Rpp32f* fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensityValue = intensityFactor[batchCount];
        Rpp32f grayValue = grayFactor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
#if __AVX2__
        __m256 pIntensity = _mm256_set1_ps(intensityValue);
        __m256 pGrayFactor = _mm256_set1_ps(grayValue);
        __m256 pConversionFactor[3];
        pConversionFactor[0] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_RED);
        pConversionFactor[1] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_GREEN);
        pConversionFactor[2] = _mm256_set1_ps(RGB_TO_GREY_WEIGHT_BLUE);
#endif
        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                         // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);            // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue)) -128.0f;
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);              // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) + 128.0f,
                                          static_cast<Rpp32f>(*srcPtrTempG) + 128.0f,
                                          static_cast<Rpp32f>(*srcPtrTempB) + 128.0f};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue)) -128.0f;
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp += 3;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                         // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f,
                                          static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f,
                                          static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue)) -128.0f;
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.B * alphaMaskFactor) + intensityMaskFactor));
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6], pFogAlphaMask[2], pFogIntensityMask[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);                           // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);                   // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);              // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity, pGrayFactor, pConversionFactor);   // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);            // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel = {static_cast<Rpp32f>(*srcPtrTempR) + 128.0f,
                                          static_cast<Rpp32f>(*srcPtrTempG) + 128.0f,
                                          static_cast<Rpp32f>(*srcPtrTempB) + 128.0f};
                    Rpp32f gray =  grayValue * ((RGB_TO_GREY_WEIGHT_RED * pixel.R) + (RGB_TO_GREY_WEIGHT_GREEN * pixel.G) + (RGB_TO_GREY_WEIGHT_BLUE * pixel.B));
                    Rpp32f oneMinusGrayValue = 1 - grayValue;
                    pixel.R = (pixel.R * oneMinusGrayValue) + gray;
                    pixel.G = (pixel.G * oneMinusGrayValue) + gray;
                    pixel.B = (pixel.B * oneMinusGrayValue) + gray;
                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue)) -128.0f;
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.R * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.G * alphaMaskFactor) + intensityMaskFactor));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((pixel.B * alphaMaskFactor) + intensityMaskFactor));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15);

            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);               // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);       // simd loads
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);                                     // simd loads
                    compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask, pIntensity);                       // fog adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);                                   // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {

                    Rpp32f alphaMaskFactor = (1 - (*fogAlphaMaskPtrTemp + intensityValue));
                    Rpp32f intensityMaskFactor = ((*fogIntensityMaskPtrTemp++) * ((*fogAlphaMaskPtrTemp++) + intensityValue)) - 128.0f;
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrTemp) + 128.0f) * alphaMaskFactor + intensityMaskFactor));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += srcDescPtr->w;
                fogIntensityMaskPtrRow += srcDescPtr->w;
            }
        }
    }

    return RPP_SUCCESS;
}

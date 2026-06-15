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
#include "rpp_cpu_rgb_hsv_conversion.hpp"
#include "rpp_cpu_simd_math.hpp"

#if __AVX2__

inline void compute_color_twist_24_host(__m256& pVecR, __m256& pVecG, __m256& pVecB,
                                        __m256* pColorTwistParams) {
    __m256 pH, pS, pV, pAdd;

    // RGB to HSV
    rgb_to_hsv(pVecR, pVecG, pVecB, pH, pS, pV, pAdd);

    // Modify Hue and Saturation
    pH = _mm256_add_ps(pH, _mm256_add_ps(pColorTwistParams[2], pAdd));  // hue += hueParam + add;
    pH = _mm256_sub_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p6, _CMP_GE_OQ),
                                         avx_p6));  // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm256_add_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p0, _CMP_LT_OQ),
                                         avx_p6));  // if (hue < 0) hue += 6.0f;
    pS = _mm256_mul_ps(pS, pColorTwistParams[3]);   // sat *= saturationParam;
    pS = _mm256_max_ps(avx_p0,
                       _mm256_min_ps(avx_p1, pS));  // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    hsv_to_rgb(pVecR, pVecG, pVecB, pH, pS, pV, pAdd);
    pVecR =
        _mm256_fmadd_ps(pVecR, pColorTwistParams[0],
                        pColorTwistParams[1]);  // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG =
        _mm256_fmadd_ps(pVecG, pColorTwistParams[0],
                        pColorTwistParams[1]);  // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB =
        _mm256_fmadd_ps(pVecB, pColorTwistParams[0],
                        pColorTwistParams[1]);  // dstPtrB = bf * brightnessParam + contrastParam;
}

#else

inline void compute_color_twist_12_host(__m128& pVecR, __m128& pVecG, __m128& pVecB,
                                        __m128* pColorTwistParams) {
    __m128 pA, pH, pS, pV, pAdd;

    // RGB to HSV
    rgb_to_hsv(pVecR, pVecG, pVecB, pH, pS, pV, pAdd);
    // Modify Hue and Saturation
    pH = _mm_add_ps(pH, _mm_add_ps(pColorTwistParams[2], pAdd));  // hue += hueParam + add;
    pH = _mm_sub_ps(pH,
                    _mm_and_ps(_mm_cmpge_ps(pH, xmm_p6), xmm_p6));  // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm_add_ps(pH, _mm_and_ps(_mm_cmplt_ps(pH, xmm_p0), xmm_p6));  // if (hue < 0) hue += 6.0f;
    pS = _mm_mul_ps(pS, pColorTwistParams[3]);                          // sat *= saturationParam;
    pS = _mm_max_ps(xmm_p0, _mm_min_ps(xmm_p1, pS));  // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    hsv_to_rgb(pVecR, pVecG, pVecB, pH, pS, pV, pAdd);
    pVecR = _mm_fmadd_ps(pVecR, pColorTwistParams[0],
                         pColorTwistParams[1]);  // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG = _mm_fmadd_ps(pVecG, pColorTwistParams[0],
                         pColorTwistParams[1]);  // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB = _mm_fmadd_ps(pVecB, pColorTwistParams[0],
                         pColorTwistParams[1]);  // dstPtrB = bf * brightnessParam + contrastParam;
}

#endif

inline void compute_color_twist_host(RpptFloatRGB* pixel, Rpp32f brightnessParam,
                                     Rpp32f contrastParam, Rpp32f hueParam,
                                     Rpp32f saturationParam) {
    // RGB to HSV

    Rpp32f hue, sat, val, add;
    Rpp32f rf = pixel->R;
    Rpp32f gf = pixel->G;
    Rpp32f bf = pixel->B;
    rgb_to_hsv(rf, gf, bf, hue, sat, val, add);

    // Modify Hue and Saturation
    hue += hueParam + add;
    if (hue >= 6.0f) hue -= 6.0f;
    if (hue < 0) hue += 6.0f;
    sat *= saturationParam;
    sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    hsv_to_rgb(rf, gf, bf, hue, sat, val, add);
    pixel->R = std::fma(rf, brightnessParam, contrastParam);
    pixel->G = std::fma(gf, brightnessParam, contrastParam);
    pixel->B = std::fma(bf, brightnessParam, contrastParam);
}

RppStatus color_twist_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                        RpptDescPtr dstDescPtr, Rpp32f* brightnessTensor,
                                        Rpp32f* contrastTensor, Rpp32f* hueTensor,
                                        Rpp32f* saturationTensor, RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType, RppLayoutParams layoutParams,
                                        rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessParam = brightnessTensor[batchCount] * 255.0f;
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueModulus = fmodf(hueTensor[batchCount], 360.0f);
        if (hueModulus < 0.0f) hueModulus += 360.0f;
        Rpp32f hueParam = hueModulus * 0.01666667f;  // 6 * 1/360
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pColorTwistParams[4];
        pColorTwistParams[0] = _mm256_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm256_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm256_set1_ps(hueParam);
        pColorTwistParams[3] = _mm256_set1_ps(saturationParam);
#else
        __m128 pColorTwistParams[4];
        pColorTwistParams[0] = _mm_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm_set1_ps(hueParam);
        pColorTwistParams[3] = _mm_set1_ps(saturationParam);
#endif

        // Color Twist with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48, p);                           // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)srcPtrTemp[0] * ONE_OVER_255;
                    pixel.G = (Rpp32f)srcPtrTemp[1] * ONE_OVER_255;
                    pixel.B = (Rpp32f)srcPtrTemp[2] * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.R)));
                    *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.G)));
                    *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.B)));

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

        // Color Twist with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);          // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48, p);  // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)*srcPtrTempR * ONE_OVER_255;
                    pixel.G = (Rpp32f)*srcPtrTempG * ONE_OVER_255;
                    pixel.B = (Rpp32f)*srcPtrTempB * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.R)));
                    dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.G)));
                    dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.B)));

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

        // Color Twist without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48, p);                           // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)srcPtrTemp[0] * ONE_OVER_255;
                    pixel.G = (Rpp32f)srcPtrTemp[1] * ONE_OVER_255;
                    pixel.B = (Rpp32f)srcPtrTemp[2] * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.R)));
                    dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.G)));
                    dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.B)));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Twist without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);          // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48, p);  // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)*srcPtrTempR * ONE_OVER_255;
                    pixel.G = (Rpp32f)*srcPtrTempG * ONE_OVER_255;
                    pixel.B = (Rpp32f)*srcPtrTempB * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.R)));
                    *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.G)));
                    *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(std::nearbyintf((pixel.B)));

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
    }

    return RPP_SUCCESS;
}

RppStatus color_twist_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp32f* brightnessTensor,
                                          Rpp32f* contrastTensor, Rpp32f* hueTensor,
                                          Rpp32f* saturationTensor, RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType, RppLayoutParams layoutParams,
                                          rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount] * ONE_OVER_255;
        Rpp32f hueModulus = fmodf(hueTensor[batchCount], 360.0f);
        if (hueModulus < 0.0f) hueModulus += 360.0f;
        Rpp32f hueParam = hueModulus * 0.01666667f;  // 6 * 1/360
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pColorTwistParams[4];
        pColorTwistParams[0] = _mm256_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm256_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm256_set1_ps(hueParam);
        pColorTwistParams[3] = _mm256_set1_ps(saturationParam);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pColorTwistParams[4];
        pColorTwistParams[0] = _mm_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm_set1_ps(hueParam);
        pColorTwistParams[3] = _mm_set1_ps(saturationParam);
#endif

        // Color Twist with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[8];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = RPPPIXELCHECKF32(pixel.B);

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

        // Color Twist with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR;
                    pixel.G = *srcPtrTempG;
                    pixel.B = *srcPtrTempB;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(pixel.B);

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

        // Color Twist without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Twist without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f32
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR;
                    pixel.G = *srcPtrTempG;
                    pixel.B = *srcPtrTempB;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = RPPPIXELCHECKF32(pixel.B);

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
    }

    return RPP_SUCCESS;
}

RppStatus color_twist_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp32f* brightnessTensor,
                                          Rpp32f* contrastTensor, Rpp32f* hueTensor,
                                          Rpp32f* saturationTensor, RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType, RppLayoutParams layoutParams,
                                          rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount] * ONE_OVER_255;
        Rpp32f hueModulus = fmodf(hueTensor[batchCount], 360.0f);
        if (hueModulus < 0.0f) hueModulus += 360.0f;
        Rpp32f hueParam = hueModulus * 0.01666667f;  // 6 * 1/360
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pColorTwistParams[4];
        pColorTwistParams[0] = _mm256_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm256_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm256_set1_ps(hueParam);
        pColorTwistParams[3] = _mm256_set1_ps(saturationParam);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128 pColorTwistParams[4];
        pColorTwistParams[0] = _mm_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm_set1_ps(hueParam);
        pColorTwistParams[3] = _mm_set1_ps(saturationParam);
#endif

        // Color Twist with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    Rpp32f srcPtrTemp_ps[13];
                    Rpp32f dstPtrTempR_ps[4], dstPtrTempG_ps[4], dstPtrTempB_ps[4];
                    for (int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];

                    __m128 p[8];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps,
                                   dstPtrTempB_ps, p);  // simd stores

                    for (int cnt = 0; cnt < vectorIncrementPerChannel; cnt++) {
                        dstPtrTempR[cnt] = (Rpp16f)dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f)dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f)dstPtrTempB_ps[cnt];
                    }
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)srcPtrTemp[0];
                    pixel.G = (Rpp32f)srcPtrTemp[1];
                    pixel.B = (Rpp32f)srcPtrTemp[2];
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp16f)RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = (Rpp16f)RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = (Rpp16f)RPPPIXELCHECKF32(pixel.B);

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

        // Color Twist with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    Rpp32f srcPtrTempR_ps[4], srcPtrTempG_ps[4], srcPtrTempB_ps[4];
                    Rpp32f dstPtrTemp_ps[13];
                    for (int cnt = 0; cnt < vectorIncrementPerChannel; cnt++) {
                        srcPtrTempR_ps[cnt] = (Rpp32f)srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f)srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f)srcPtrTempB[cnt];
                    }
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps,
                                  srcPtrTempB_ps, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps,
                                   p);  // simd stores
                    for (int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)*srcPtrTempR;
                    pixel.G = (Rpp32f)*srcPtrTempG;
                    pixel.B = (Rpp32f)*srcPtrTempB;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp16f)RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = (Rpp16f)RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = (Rpp16f)RPPPIXELCHECKF32(pixel.B);

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

        // Color Twist without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    Rpp32f srcPtrTemp_ps[13];
                    Rpp32f dstPtrTemp_ps[13];
                    for (int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps,
                                   p);  // simd stores

                    for (int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)srcPtrTemp[0];
                    pixel.G = (Rpp32f)srcPtrTemp[1];
                    pixel.B = (Rpp32f)srcPtrTemp[2];
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp16f)RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = (Rpp16f)RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = (Rpp16f)RPPPIXELCHECKF32(pixel.B);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Twist without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_color_twist_24_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    Rpp32f srcPtrTempR_ps[4], srcPtrTempG_ps[4], srcPtrTempB_ps[4];
                    Rpp32f dstPtrTempR_ps[4], dstPtrTempG_ps[4], dstPtrTempB_ps[4];
                    for (int cnt = 0; cnt < vectorIncrementPerChannel; cnt++) {
                        srcPtrTempR_ps[cnt] = (Rpp32f)srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f)srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f)srcPtrTempB[cnt];
                    }

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps,
                                  srcPtrTempB_ps, p);  // simd loads
                    compute_color_twist_12_host(p[0], p[1], p[2],
                                                pColorTwistParams);  // color_twist adjustment
                    // Boundary checks for f16
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps,
                                   dstPtrTempB_ps, p);  // simd stores

                    for (int cnt = 0; cnt < vectorIncrementPerChannel; cnt++) {
                        dstPtrTempR[cnt] = (Rpp16f)dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f)dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f)dstPtrTempB_ps[cnt];
                    }
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (Rpp32f)*srcPtrTempR;
                    pixel.G = (Rpp32f)*srcPtrTempG;
                    pixel.B = (Rpp32f)*srcPtrTempB;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp16f)RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = (Rpp16f)RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = (Rpp16f)RPPPIXELCHECKF32(pixel.B);

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
    }

    return RPP_SUCCESS;
}

RppStatus color_twist_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                        RpptDescPtr dstDescPtr, Rpp32f* brightnessTensor,
                                        Rpp32f* contrastTensor, Rpp32f* hueTensor,
                                        Rpp32f* saturationTensor, RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType, RppLayoutParams layoutParams,
                                        rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessParam = brightnessTensor[batchCount] * 255.0f;
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueModulus = fmodf(hueTensor[batchCount], 360.0f);
        if (hueModulus < 0.0f) hueModulus += 360.0f;
        Rpp32f hueParam = hueModulus * 0.01666667f;  // 6 * 1/360
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

#if __AVX2__
        __m256 pColorTwistParams[4];
        pColorTwistParams[0] = _mm256_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm256_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm256_set1_ps(hueParam);
        pColorTwistParams[3] = _mm256_set1_ps(saturationParam);
#else
        __m128 pColorTwistParams[4];
        pColorTwistParams[0] = _mm_set1_ps(brightnessParam);
        pColorTwistParams[1] = _mm_set1_ps(contrastParam);
        pColorTwistParams[2] = _mm_set1_ps(hueParam);
        pColorTwistParams[3] = _mm_set1_ps(saturationParam);
#endif

        // Color Twist with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48, p);                           // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = ((Rpp32f)srcPtrTemp[0] + 128.0f) * ONE_OVER_255;
                    pixel.G = ((Rpp32f)srcPtrTemp[1] + 128.0f) * ONE_OVER_255;
                    pixel.B = ((Rpp32f)srcPtrTemp[2] + 128.0f) * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp8s)RPPPIXELCHECKI8(pixel.R - 128.0f);
                    *dstPtrTempG = (Rpp8s)RPPPIXELCHECKI8(pixel.G - 128.0f);
                    *dstPtrTempB = (Rpp8s)RPPPIXELCHECKI8(pixel.B - 128.0f);

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

        // Color Twist with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);          // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48, p);  // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = ((Rpp32f)*srcPtrTempR + 128.0f) * ONE_OVER_255;
                    pixel.G = ((Rpp32f)*srcPtrTempG + 128.0f) * ONE_OVER_255;
                    pixel.B = ((Rpp32f)*srcPtrTempB + 128.0f) * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8(pixel.R - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8(pixel.G - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8(pixel.B - 128.0f);

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

        // Color Twist without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48, p);                           // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = ((Rpp32f)srcPtrTemp[0] + 128.0f) * ONE_OVER_255;
                    pixel.G = ((Rpp32f)srcPtrTemp[1] + 128.0f) * ONE_OVER_255;
                    pixel.B = ((Rpp32f)srcPtrTemp[2] + 128.0f) * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8(pixel.R - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8(pixel.G - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8(pixel.B - 128.0f);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Twist without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);          // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);  // simd normalize
                    compute_color_twist_24_host(p[0], p[2], p[4],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_24_host(p[1], p[3], p[5],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48, p);  // simd normalize
                    compute_color_twist_12_host(p[0], p[4], p[8],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[1], p[5], p[9],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[2], p[6], p[10],
                                                pColorTwistParams);  // color_twist adjustment
                    compute_color_twist_12_host(p[3], p[7], p[11],
                                                pColorTwistParams);  // color_twist adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = ((Rpp32f)*srcPtrTempR + 128.0f) * ONE_OVER_255;
                    pixel.G = ((Rpp32f)*srcPtrTempG + 128.0f) * ONE_OVER_255;
                    pixel.B = ((Rpp32f)*srcPtrTempB + 128.0f) * ONE_OVER_255;
                    compute_color_twist_host(&pixel, brightnessParam, contrastParam, hueParam,
                                             saturationParam);
                    *dstPtrTempR = (Rpp8s)RPPPIXELCHECKI8(pixel.R - 128.0f);
                    *dstPtrTempG = (Rpp8s)RPPPIXELCHECKI8(pixel.G - 128.0f);
                    *dstPtrTempB = (Rpp8s)RPPPIXELCHECKI8(pixel.B - 128.0f);

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
    }

    return RPP_SUCCESS;
}

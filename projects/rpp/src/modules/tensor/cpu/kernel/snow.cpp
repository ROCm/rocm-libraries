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

#include "host_tensor_executors.hpp"
#include "rpp_cpu_simd_math.hpp"

static const Rpp32f SNOW_HUE_LOWER_BOUND = 0.514f;      // Lower bound of hue range to exclude
static const Rpp32f SNOW_HUE_UPPER_BOUND = 0.63f;       // Upper bound of hue range to exclude
static const Rpp32f SNOW_SAT_THRESHOLD = 0.196f;        // Saturation threshold for color filtering
static const Rpp32f SNOW_LIGHTNESS_THRESHOLD = 0.196f;  // Lightness threshold for color filtering

inline void compute_snow_host_gray(Rpp32f& pixel, Rpp32f brightnessCoefficient,
                                   Rpp32f snowCoefficient, Rpp32s darkMode) {
    const Rpp32f lower_threshold = 0.0f;
    const Rpp32f upper_threshold = 0.39215686f;
    const Rpp32f thresholdDiff = 0.39215686f;  // upper_threshold - lower_threshold
    const Rpp32f brightnessFactor = 2.5f;

    // Dark mode enhancement
    if (darkMode == 1 && pixel >= lower_threshold && pixel <= upper_threshold)
        pixel *= std::fmaf(-pixel / thresholdDiff, brightnessFactor - 1.0f, brightnessFactor);

    // Snow brightness
    if (pixel <= snowCoefficient) pixel *= brightnessCoefficient;
}

inline void compute_snow_host(RpptFloatRGB* pixel, Rpp32f brightnessCoefficient,
                              Rpp32f snowCoefficient, Rpp32s darkMode) {
    // RGB to HSL
    Rpp32f hue, sat, l, add;
    Rpp32f rf, gf, bf, cmax, cmin, delta;
    const Rpp32f lower_threshold = 0.0f;
    const Rpp32f upper_threshold = 0.39215686f;
    const Rpp32f thresholdDiff = 0.39215686f;  // upper_threshold - lower_threshold
    const Rpp32f brightnessFactor = 2.5f;
    rf = pixel->R;
    gf = pixel->G;
    bf = pixel->B;
    cmax = RPPMAX3(rf, gf, bf);
    cmin = RPPMIN3(rf, gf, bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    l = (cmax + cmin) * 0.5;
    if ((delta != 0)) {
        if (l <= 0.5)
            sat = delta / (cmax + cmin);
        else
            sat = delta / (2.0f - (cmax + cmin));

        if (cmax == rf) {
            hue = gf - bf;
            add = 0.0f;
        } else if (cmax == gf) {
            hue = bf - rf;
            add = 2.0f;
        } else {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
        hue += add;
        hue /= 6.0f;
    }
    // Modify Lightness
    if (l >= lower_threshold && l <= upper_threshold && darkMode == 1)
        l = l * std::fmaf(-l / thresholdDiff, brightnessFactor - 1.0f, brightnessFactor);

    if (l <= snowCoefficient && !((hue >= SNOW_HUE_LOWER_BOUND && hue <= SNOW_HUE_UPPER_BOUND) &&
                                  (sat >= SNOW_SAT_THRESHOLD) && (l >= SNOW_LIGHTNESS_THRESHOLD)))
        l = l * brightnessCoefficient;

    // HSL to RGB with brightness/contrast adjustment
    Rpp32f hueCoefficient[3];
    hueCoefficient[0] = 6.0f * (hue - TWO_OVER_3);
    hueCoefficient[1] = 0.0f;
    hueCoefficient[2] = 6.0f * (1.0f - hue);
    if (hue < TWO_OVER_3) {
        hueCoefficient[0] = 0.0f;
        hueCoefficient[1] = 6.0f * (TWO_OVER_3 - hue);
        hueCoefficient[2] = 6.0f * (hue - ONE_OVER_3);
    }
    if (hue < ONE_OVER_3) {
        hueCoefficient[0] = 6.0f * (ONE_OVER_3 - hue);
        hueCoefficient[1] = 6.0f * hue;
        hueCoefficient[2] = 0.0f;
    }
    hueCoefficient[0] = RPPMIN2(hueCoefficient[0], 1.0f);
    hueCoefficient[1] = RPPMIN2(hueCoefficient[1], 1.0f);
    hueCoefficient[2] = RPPMIN2(hueCoefficient[2], 1.0f);

    Rpp32f sat2 = 2.0f * sat;
    Rpp32f satInv = 1.0f - sat;
    Rpp32f lumInv = 1.0f - l;
    Rpp32f lum2m1 = (2.0f * l) - 1.0f;
    hueCoefficient[0] = (sat2 * hueCoefficient[0]) + satInv;
    hueCoefficient[1] = (sat2 * hueCoefficient[1]) + satInv;
    hueCoefficient[2] = (sat2 * hueCoefficient[2]) + satInv;
    if (l >= 0.5f) {
        hueCoefficient[0] = (lumInv * hueCoefficient[0]) + lum2m1;
        hueCoefficient[1] = (lumInv * hueCoefficient[1]) + lum2m1;
        hueCoefficient[2] = (lumInv * hueCoefficient[2]) + lum2m1;
    } else {
        hueCoefficient[0] = hueCoefficient[0] * l;
        hueCoefficient[1] = hueCoefficient[1] * l;
        hueCoefficient[2] = hueCoefficient[2] * l;
    }
    pixel->R = hueCoefficient[0];
    pixel->G = hueCoefficient[1];
    pixel->B = hueCoefficient[2];
}

inline void compute_snow_24_host(__m256& pVecR, __m256& pVecG, __m256& pVecB, __m256* pSnowParams) {
    __m256 pH, pS, pL, pCmax, pCmin, pDelta, pAdd;
    __m256 pMask[4];
    __m256 pHueCoefficient[3];

    // RGB to HSL
    pCmax = _mm256_max_ps(pVecR, _mm256_max_ps(pVecG, pVecB));  // cmax = RPPMAX3(rf, gf, bf);
    pCmin = _mm256_min_ps(pVecR, _mm256_min_ps(pVecG, pVecB));  // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm256_sub_ps(pCmax, pCmin);                       // delta = cmax - cmin;
    pH = avx_p0;                                                // hue = 0.0f;
    pS = avx_p0;                                                // sat = 0.0f;
    pAdd = avx_p0;                                              // add = 0.0f;
    pL = _mm256_mul_ps(_mm256_add_ps(pCmax, pCmin),
                       _mm256_set1_ps(0.5f));               // l = (cmax + cmin) * 0.5;
    pMask[0] = _mm256_cmp_ps(pDelta, avx_p0, _CMP_NEQ_OQ);  // if ((delta != 0)) {
    pMask[1] = _mm256_cmp_ps(pL, _mm256_set1_ps(0.5f),
                             _CMP_LE_OQ);          //     Temporarily store l <= 0.5 comparison
    pMask[3] = _mm256_and_ps(pMask[0], pMask[1]);  //     if (l <= 0.5)  {
    pS = _mm256_and_ps(
        pMask[3],
        _mm256_div_ps(pDelta,
                      _mm256_add_ps(pCmax, pCmin)));  //          sat = delta / (cmax + cmin); }
    pMask[3] = _mm256_andnot_ps(pMask[1], pMask[0]);  //      else {
    pS = _mm256_blendv_ps(
        pS, _mm256_div_ps(pDelta, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(pCmax, pCmin))),
        pMask[3]);  // sat = delta / (2.0f - (cmax + cmin));  }
    pMask[1] =
        _mm256_cmp_ps(pCmax, pVecR, _CMP_EQ_OQ);   //     Temporarily store cmax == rf comparison
    pMask[2] = _mm256_and_ps(pMask[0], pMask[1]);  //     if (cmax == rf)
    pH = _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecG, pVecB));  //         hue = gf - bf;
    pAdd = _mm256_and_ps(pMask[2], avx_p0);                     //         add = 0.0f;
    pMask[3] =
        _mm256_cmp_ps(pCmax, pVecG, _CMP_EQ_OQ);      //     Temporarily store cmax == gf comparison
    pMask[2] = _mm256_andnot_ps(pMask[1], pMask[3]);  //     else if (cmax == gf)
    pH = _mm256_blendv_ps(pH, _mm256_sub_ps(pVecB, pVecR), pMask[2]);  //         hue = bf - rf;
    pAdd = _mm256_blendv_ps(pAdd, avx_p2, pMask[2]);                   //         add = 2.0f;
    pMask[3] = _mm256_andnot_ps(pMask[3], _mm256_andnot_ps(pMask[1], pMask[0]));  //     else
    pH = _mm256_blendv_ps(pH, _mm256_sub_ps(pVecR, pVecG), pMask[3]);  //         hue = rf - gf;
    pAdd = _mm256_blendv_ps(pAdd, avx_p4, pMask[3]);                   //         add = 4.0f;
    pH = _mm256_blendv_ps(pH, _mm256_div_ps(pH, pDelta), pMask[0]);    //     hue /= delta; }
    pH = _mm256_add_ps(pH, pAdd);
    pH = _mm256_mul_ps(pH, avx_p1op6);

    // Modify Lightness
    __m256 pLowerThreshold, pUpperThreshold, pDiffThreshold, pBrightnessFactor;
    pLowerThreshold = avx_p0;
    pUpperThreshold = _mm256_set1_ps(0.39215686f);
    pBrightnessFactor = _mm256_set1_ps(2.5f);
    pDiffThreshold = _mm256_sub_ps(pUpperThreshold, pLowerThreshold);
    pMask[0] = _mm256_cmp_ps(pL, pLowerThreshold,
                             _CMP_GE_OQ);  // Temporarily store l >= lower_threshold comparison
    pMask[1] = _mm256_cmp_ps(pL, pUpperThreshold,
                             _CMP_LE_OQ);  // Temporarily store l <= upper_threshold comparison
    pMask[0] = _mm256_and_ps(
        pMask[0],
        pMask[1]);  // Temporarily store (l >= lower_threshold && l <= upper_threshold) comparision
    pMask[1] = _mm256_cmp_ps(pSnowParams[2], avx_p1,
                             _CMP_EQ_OQ);  // Temporarily store darkmode == 1.0f comparison
    pMask[3] = _mm256_and_ps(
        pMask[0], pMask[1]);  // if(l >= lower_threshold && l <= upper_threshold && darkMode ==1)
    __m256 pLDivDiff = _mm256_div_ps(pL, pDiffThreshold);  // l / thresholdDiff
    __m256 pBrightnessScale = _mm256_fmsub_ps(
        _mm256_sub_ps(avx_p1, pBrightnessFactor), pLDivDiff,
        pBrightnessFactor);  // brightnessFactor - (brightnessFactor - 1) * l/thresholdDiff
    pL = _mm256_blendv_ps(pL, _mm256_mul_ps(pL, pBrightnessScale), pMask[3]);  // l = l * scale
    pMask[0] =
        _mm256_cmp_ps(pH, _mm256_set1_ps(SNOW_HUE_LOWER_BOUND),
                      _CMP_GE_OQ);  // Temporarily store hue >= SNOW_HUE_LOWER_BOUND comparision
    pMask[1] =
        _mm256_cmp_ps(pH, _mm256_set1_ps(SNOW_HUE_UPPER_BOUND),
                      _CMP_LE_OQ);  // Temporarily store hue <= SNOW_HUE_UPPER_BOUND comparison
    pMask[0] = _mm256_and_ps(pMask[0], pMask[1]);  // Temporarily store (hue >= SNOW_HUE_LOWER_BOUND
                                                   // && hue <= SNOW_HUE_UPPER_BOUND) comparison
    pMask[1] =
        _mm256_cmp_ps(pS, _mm256_set1_ps(SNOW_SAT_THRESHOLD),
                      _CMP_GE_OQ);  // Temporarily store (sat >= SNOW_SAT_THRESHOLD) comparison
    pMask[0] = _mm256_and_ps(
        pMask[0], pMask[1]);  // Temporarily store (hue >= SNOW_HUE_LOWER_BOUND && hue <=
                              // SNOW_HUE_UPPER_BOUND) && (sat >= SNOW_SAT_THRESHOLD) comparison
    pMask[1] =
        _mm256_cmp_ps(pL, _mm256_set1_ps(SNOW_LIGHTNESS_THRESHOLD),
                      _CMP_GE_OQ);  // Temporarily store (l >= SNOW_LIGHTNESS_THRESHOLD) comparison
    pMask[0] = _mm256_and_ps(pMask[0],
                             pMask[1]);  // Temporarily store (hue >= SNOW_HUE_LOWER_BOUND && hue <=
                                         // SNOW_HUE_UPPER_BOUND) && (sat >= SNOW_SAT_THRESHOLD) &&
                                         // (l >= SNOW_LIGHTNESS_THRESHOLD) comparison
    pMask[1] = _mm256_cmp_ps(pL, pSnowParams[1],
                             _CMP_LE_OQ);  // Temporarily store (l <= *snowCoefficient) comparison
    pMask[0] = _mm256_andnot_ps(
        pMask[0], pMask[1]);  // if(l <= *snowCoefficient && !((hue >= SNOW_HUE_LOWER_BOUND && hue
                              // <= SNOW_HUE_UPPER_BOUND) && (sat >= SNOW_SAT_THRESHOLD) && (l >=
                              // SNOW_LIGHTNESS_THRESHOLD)))
    pL = _mm256_blendv_ps(pL, _mm256_mul_ps(pL, pSnowParams[0]),
                          pMask[0]);  //     l = l * (*brightnessCoefficient);

    // HSL to RGB with brightness/contrast adjustment
    pHueCoefficient[0] = _mm256_mul_ps(
        _mm256_set1_ps(6.0f),
        _mm256_sub_ps(pH, avx_p2op3));  // hueCoefficient[0] = 6.0f * (hue - 2.0f/3.0f);
    pHueCoefficient[1] = avx_p0;        // hueCoefficient[1] = 0.0f;
    pHueCoefficient[2] =
        _mm256_mul_ps(_mm256_set1_ps(6.0f),
                      _mm256_sub_ps(avx_p1, pH));  // hueCoefficient[2] = 6.0f * (1.0f - hue);
    pMask[0] = _mm256_cmp_ps(pH, avx_p2op3, _CMP_LT_OQ);  // if(hue < 2.0f/3.0f){
    pHueCoefficient[0] =
        _mm256_blendv_ps(pHueCoefficient[0], avx_p0, pMask[0]);  //     hueCoefficient[0] = 0.0f;
    pHueCoefficient[1] =
        _mm256_blendv_ps(pHueCoefficient[1], _mm256_mul_ps(avx_p6, _mm256_sub_ps(avx_p2op3, pH)),
                         pMask[0]);  //    hueCoefficient[1] = 6.0f * ((2/3) - hue);
    pHueCoefficient[2] =
        _mm256_blendv_ps(pHueCoefficient[2], _mm256_mul_ps(avx_p6, _mm256_sub_ps(pH, avx_p1op3)),
                         pMask[0]);  //    hueCoefficient[2] = 6.0f * (hue - (1/3)); }

    pMask[0] = _mm256_cmp_ps(pH, avx_p1op3, _CMP_LT_OQ);  // if(hue < 1.0f/3.0f) {
    pHueCoefficient[0] =
        _mm256_blendv_ps(pHueCoefficient[0], _mm256_mul_ps(avx_p6, _mm256_sub_ps(avx_p1op3, pH)),
                         pMask[0]);  //    hueCoefficient[0] = 6.0f * ((1/3) -hue );
    pHueCoefficient[1] = _mm256_blendv_ps(pHueCoefficient[1], _mm256_mul_ps(avx_p6, pH),
                                          pMask[0]);  //    hueCoefficient[1] = 6.0f * hue;
    pHueCoefficient[2] =
        _mm256_blendv_ps(pHueCoefficient[2], avx_p0, pMask[0]);  //    hueCoefficient[2] = 0.0f;

    pHueCoefficient[0] = _mm256_min_ps(
        pHueCoefficient[0], avx_p1);  // hueCoefficient[0] = RPPMIN2(hueCoefficient[0], 1.0f)
    pHueCoefficient[1] = _mm256_min_ps(
        pHueCoefficient[1], avx_p1);  // hueCoefficient[1] = RPPMIN2(hueCoefficient[1], 1.0f);
    pHueCoefficient[2] = _mm256_min_ps(
        pHueCoefficient[2], avx_p1);  // hueCoefficient[2] = RPPMIN2(hueCoefficient[2], 1.0f);

    __m256 pSat2, pSatinv, pLuminv, pLum2m1;
    pSat2 = _mm256_mul_ps(avx_p2, pS);              // sat2 = 2.0f * sat;
    pSatinv = _mm256_sub_ps(avx_p1, pS);            // satInv = 1.0f - sat;
    pLuminv = _mm256_sub_ps(avx_p1, pL);            // lumInv = 1.0f - l;
    pLum2m1 = _mm256_fmsub_ps(avx_p2, pL, avx_p1);  // lum2m1 = (2.0f * l) - 1.0f;

    pHueCoefficient[0] =
        _mm256_fmadd_ps(pSat2, pHueCoefficient[0],
                        pSatinv);  // hueCoefficient[0] = (sat2 * hueCoefficient[0]) + satInv;
    pHueCoefficient[1] =
        _mm256_fmadd_ps(pSat2, pHueCoefficient[1],
                        pSatinv);  // hueCoefficient[1] = (sat2 * hueCoefficient[1]) + satInv;
    pHueCoefficient[2] =
        _mm256_fmadd_ps(pSat2, pHueCoefficient[2],
                        pSatinv);  // hueCoefficient[2] = (sat2 * hueCoefficient[2]) + satInv;

    pMask[0] = _mm256_cmp_ps(pL, _mm256_set1_ps(0.5f),
                             _CMP_GE_OQ);  // Temporarily store (l >= 0.5f) comparison
    pVecR = _mm256_blendv_ps(
        _mm256_mul_ps(pL, pHueCoefficient[0]),
        _mm256_fmadd_ps(pLuminv, pHueCoefficient[0], pLum2m1),
        pMask[0]);  // if(l >= 0.5f) {hueCoefficient[0] = (lumInv * hueCoefficient[0]) + lum2m1; }
                    // else {hueCoefficient[0] = hueCoefficient[0] * l;}
    pVecG = _mm256_blendv_ps(
        _mm256_mul_ps(pL, pHueCoefficient[1]),
        _mm256_fmadd_ps(pLuminv, pHueCoefficient[1], pLum2m1),
        pMask[0]);  // if(l >= 0.5f) {hueCoefficient[1] = (lumInv * hueCoefficient[1]) + lum2m1; }
                    // else {hueCoefficient[1] = hueCoefficient[1] * l;}
    pVecB = _mm256_blendv_ps(
        _mm256_mul_ps(pL, pHueCoefficient[2]),
        _mm256_fmadd_ps(pLuminv, pHueCoefficient[2], pLum2m1),
        pMask[0]);  // if(l >= 0.5f) {hueCoefficient[2] = (lumInv * hueCoefficient[2]) + lum2m1; }
                    // else {hueCoefficient[2] = hueCoefficient[2] * l;}
}

inline void compute_snow_8_host(__m256* p, __m256* pSnowParams) {
    __m256 pL = p[0];
    __m256 pMask[4];

    // Modify Hue and Saturation
    __m256 pLowerThreshold, pUpperThreshold, pDiffThreshold, pBrightnessFactor;
    pLowerThreshold = avx_p0;
    pUpperThreshold = _mm256_set1_ps(0.39215686f);
    pBrightnessFactor = _mm256_set1_ps(2.5f);
    pDiffThreshold = _mm256_sub_ps(pUpperThreshold, pLowerThreshold);
    pMask[0] = _mm256_cmp_ps(pL, pLowerThreshold,
                             _CMP_GE_OQ);  // Temporarily store (l >= lower_threshold) comparison
    pMask[1] = _mm256_cmp_ps(pL, pUpperThreshold,
                             _CMP_LE_OQ);  // Temporarily store l <= upper_threshold comparison
    pMask[0] = _mm256_and_ps(
        pMask[0],
        pMask[1]);  // Temporarily store (l >= lower_threshold && l <= upper_threshold) comparison
    pMask[1] = _mm256_cmp_ps(pSnowParams[2], avx_p1,
                             _CMP_EQ_OQ);  // Temporarily store darkmode == 1.0f comparison
    pMask[3] = _mm256_and_ps(
        pMask[0], pMask[1]);  // if(l >= lower_threshold && l <= upper_threshold && darkMode ==1)
    __m256 pLDivDiff = _mm256_div_ps(pL, pDiffThreshold);  // l / thresholdDiff
    __m256 pBrightnessScale = _mm256_fmsub_ps(
        _mm256_sub_ps(avx_p1, pBrightnessFactor), pLDivDiff,
        pBrightnessFactor);  // brightnessFactor - (brightnessFactor - 1) * l/thresholdDiff
    pL = _mm256_blendv_ps(pL, _mm256_mul_ps(pL, pBrightnessScale), pMask[3]);  // l = l * scale
    pMask[0] = _mm256_cmp_ps(pL, pSnowParams[1],
                             _CMP_LE_OQ);  // Temporarily store (l <= *snowCoefficient) comparison
    p[0] = _mm256_blendv_ps(pL, _mm256_mul_ps(pL, pSnowParams[0]),
                            pMask[0]);  // l = l * (*brightnessCoefficient);
}

RppStatus snow_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32f* brightnessCoefficientTensor,
                                 Rpp32f* snowThresholdTensor, Rpp32s* darkModeTensor,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = std::fmaf(snowThresholdTensor[batchCount], 0.5f, 0.333333333f);
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<Rpp32f>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    *dstPtrTempR++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    *dstPtrTempG++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    *dstPtrTempB++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR++) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG++) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB++) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR++) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG++) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB++) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    *dstPtrTempR++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    *dstPtrTempG++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    *dstPtrTempB++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
            alignedLength = (bufferLength & ~15);
#endif
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    rpp_simd_load(rpp_normalize16_avx, p);                     // simd normalize
                    compute_snow_8_host(&p[0], pSnowParams);                   // snow adjustment
                    compute_snow_8_host(&p[1], pSnowParams);                   // snow adjustment
                    rpp_multiply16_constant(p, avx_p255);                      // simd denormalize
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f pixel;
                    pixel = static_cast<Rpp32f>(*srcPtrTemp++) * ONE_OVER_255;
                    compute_snow_host_gray(pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel *= 255.0f;
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel)));
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus snow_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* brightnessCoefficientTensor,
                                   Rpp32f* snowThresholdTensor, Rpp32s* darkModeTensor,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = std::fmaf(snowThresholdTensor[batchCount], 0.5f, 0.333333333f);
        Rpp32s darkMode = darkModeTensor[batchCount];

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
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<Rpp32f>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR++ = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR++;
                    pixel.G = *srcPtrTempG++;
                    pixel.B = *srcPtrTempB++;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(pixel.B);

                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
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

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR++;
                    pixel.G = *srcPtrTempG++;
                    pixel.B = *srcPtrTempB++;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR++ = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(pixel.B);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
            alignedLength = (bufferLength & ~7);
#endif
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);  // simd loads
                    compute_snow_8_host(&p, pSnowParams);                     // snow adjustment
                    rpp_pixel_check_0to1(&p, 1);
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f pixel;
                    pixel = *srcPtrTemp++;
                    compute_snow_host_gray(pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTemp++ = RPPPIXELCHECKF32(pixel);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus snow_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* brightnessCoefficientTensor,
                                   Rpp32f* snowThresholdTensor, Rpp32s* darkModeTensor,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = std::fmaf(snowThresholdTensor[batchCount], 0.5f, 0.333333333f);
        Rpp32s darkMode = darkModeTensor[batchCount];

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
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<Rpp32f>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB++);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);  // snow adjustment
                    rpp_pixel_check_0to1(p, 3);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR++);
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG++);
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB++);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
            alignedLength = (bufferLength & ~7);
#endif
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p;
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, &p);  // simd loads
                    compute_snow_8_host(&p, pSnowParams);                     // snow adjustment
                    rpp_pixel_check_0to1(&p, 1);
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f pixel;
                    pixel = static_cast<Rpp32f>(*srcPtrTemp++);
                    compute_snow_host_gray(pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel));
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus snow_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32f* brightnessCoefficientTensor,
                                 Rpp32f* snowThresholdTensor, Rpp32s* darkModeTensor,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = std::fmaf(snowThresholdTensor[batchCount], 0.5f, 0.333333333f);
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<Rpp32f>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(*srcPtrTempR++) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(*srcPtrTempG++) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(*srcPtrTempB++) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                        // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);  // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);  // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                 // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(*srcPtrTempR++) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(*srcPtrTempG++) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(*srcPtrTempB++) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
#if __AVX2__
            alignedLength = (bufferLength & ~15);
#endif
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    rpp_simd_load(rpp_normalize16_avx, p);                     // simd normalize
                    compute_snow_8_host(&p[0], pSnowParams);                   // snow adjustment
                    compute_snow_8_host(&p[1], pSnowParams);                   // snow adjustment
                    rpp_multiply16_constant(p, avx_p255);                      // simd denormalize
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f pixel;
                    pixel = (static_cast<Rpp32f>(*srcPtrTemp++) + 128.0f) * ONE_OVER_255;
                    compute_snow_host_gray(pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel *= 255.0f;
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel - 128.0f));
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

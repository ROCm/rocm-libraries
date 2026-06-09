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

#include "rpp_cpu_common.hpp"

#if __AVX2__

// Converts RGB to HSV color space using AVX vectorization, processes 8 pixels simultaneously
inline void rgb_to_hsv(__m256 &pVecR, __m256 &pVecG, __m256 &pVecB, __m256 &pH, __m256 &pS, __m256 &pV, __m256 &pAdd)
{
    __m256 pMask[4], pDelta;
    // RGB to HSV
    pV = _mm256_max_ps(pVecR, _mm256_max_ps(pVecG, pVecB));                                                         // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm256_min_ps(pVecR, _mm256_min_ps(pVecG, pVecB));                                                         // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm256_sub_ps(pV, pS);                                                                                 // delta = cmax - cmin;
    pMask[0] = _mm256_and_ps(_mm256_cmp_ps(pDelta, avx_p0, _CMP_NEQ_OQ), _mm256_cmp_ps(pV, avx_p0, _CMP_NEQ_OQ));   // if ((delta != 0) && (cmax != 0)) {
    pS = _mm256_div_ps(_mm256_and_ps(pMask[0], pDelta), pV);                                                        //     sat = delta / cmax;
    pMask[1] = _mm256_cmp_ps(pV, pVecR, _CMP_EQ_OQ);                                                                //     Temporarily store cmax == rf comparison
    pMask[2] = _mm256_and_ps(pMask[0], pMask[1]);                                                                   //     if (cmax == rf)
    pH = _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecG, pVecB));                                                      //         hue = gf - bf;
    pAdd = _mm256_and_ps(pMask[2], avx_p0);                                                                         //         add = 0.0f;
    pMask[3] = _mm256_cmp_ps(pV, pVecG, _CMP_EQ_OQ);                                                                //     Temporarily store cmax == gf comparison
    pMask[2] = _mm256_andnot_ps(pMask[1], pMask[3]);                                                                //     else if (cmax == gf)
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pH), _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecB, pVecR)));        //         hue = bf - rf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pAdd), _mm256_and_ps(pMask[2], avx_p2));                         //         add = 2.0f;
    pMask[3] = _mm256_andnot_ps(pMask[3], _mm256_andnot_ps(pMask[1], pMask[0]));                                    //     else
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pH), _mm256_and_ps(pMask[3], _mm256_sub_ps(pVecR, pVecG)));        //         hue = rf - gf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pAdd), _mm256_and_ps(pMask[3], avx_p4));                         //         add = 4.0f;
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pH), _mm256_and_ps(pMask[0], _mm256_div_ps(pH, pDelta)));          //     hue /= delta; }}
}

// Converts HSV to RGB color space using AVX vectorization, processes 8 pixels simultaneously
inline void hsv_to_rgb(__m256 &pVecR, __m256 &pVecG, __m256 &pVecB, __m256 &pH, __m256 &pS, __m256 &pV, __m256 &pAdd)
{
    __m256 pMask[4], pIntH, pA;
    __m256i pxIntH;
    // HSV to RGB with brightness/contrast adjustment
    pIntH = _mm256_floor_ps(pH);                                                                                    // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm256_cvtps_epi32(pIntH);                                                                             // Convert to epi32
    pH = _mm256_sub_ps(pH, pIntH);                                                                                  // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm256_mul_ps(pV, pS);                                                                                     // Rpp32f vsat = v * sat;
    pAdd = _mm256_mul_ps(pS, pH);                                                                                   // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm256_sub_ps(pV, pS);                                                                                     // Rpp32f p = v - vsat;
    pH = _mm256_sub_ps(pV, pAdd);                                                                                   // Rpp32f q = v - vsatf;
    pS = _mm256_add_ps(pA, pAdd);                                                                                   // Rpp32f t = v - vsat + vsatf;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px0));                                            // switch (hueIntegerPart) {case 0:
    pVecR = _mm256_and_ps(pMask[0], pV);                                                                            //     rf = v;
    pVecG = _mm256_and_ps(pMask[0], pS);                                                                            //     gf = t;
    pVecB = _mm256_and_ps(pMask[0], pA);                                                                            //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px1));                                            // case 1:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pH));                           //     rf = q;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                           //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pA));                           //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px2));                                            // case 2:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                           //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                           //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pS));                           //     bf = t; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px3));                                            // case 3:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                           //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pH));                           //     gf = q;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                           //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px4));                                            // case 4:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pS));                           //     rf = t;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                           //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                           //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px5));                                            // case 5:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pV));                           //     rf = v;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                           //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pH));                           //     bf = q; break;}
}

#else

// Converts RGB to HSV color space using SSE vectorization, processes 4 pixels simultaneously
inline void rgb_to_hsv(__m128 &pVecR, __m128 &pVecG, __m128 &pVecB, __m128 &pH, __m128 &pS, __m128 &pV, __m128 &pAdd)
{
    __m128 pMask[4], pDelta;
    // Convert RGB to HSV
    pV = _mm_max_ps(pVecR, _mm_max_ps(pVecG, pVecB));                                                               // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm_min_ps(pVecR, _mm_min_ps(pVecG, pVecB));                                                               // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm_sub_ps(pV, pS);                                                                                    // delta = cmax - cmin;
    pMask[0] = _mm_and_ps(_mm_cmpneq_ps(pDelta, xmm_p0), _mm_cmpneq_ps(pV, xmm_p0));                                // if ((delta != 0) && (cmax != 0)) {
    pS = _mm_div_ps(_mm_and_ps(pMask[0], pDelta), pV);                                                              //     sat = delta / cmax;
    pMask[1] = _mm_cmpeq_ps(pV, pVecR);                                                                             //     Temporarily store cmax == rf comparison
    pMask[2] = _mm_and_ps(pMask[0], pMask[1]);                                                                      //     if (cmax == rf)
    pH = _mm_and_ps(pMask[2], _mm_sub_ps(pVecG, pVecB));                                                            //         hue = gf - bf;
    pAdd = _mm_and_ps(pMask[2], xmm_p0);                                                                            //         add = 0.0f;
    pMask[3] = _mm_cmpeq_ps(pV, pVecG);                                                                             //     Temporarily store cmax == gf comparison
    pMask[2] = _mm_andnot_ps(pMask[1], pMask[3]);                                                                   //     else if (cmax == gf)
    pH = _mm_or_ps(_mm_andnot_ps(pMask[2], pH), _mm_and_ps(pMask[2], _mm_sub_ps(pVecB, pVecR)));                    //         hue = bf - rf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[2], pAdd), _mm_and_ps(pMask[2], xmm_p2));                                  //         add = 2.0f;
    pMask[3] = _mm_andnot_ps(pMask[3], _mm_andnot_ps(pMask[1], pMask[0]));                                          //     else
    pH = _mm_or_ps(_mm_andnot_ps(pMask[3], pH), _mm_and_ps(pMask[3], _mm_sub_ps(pVecR, pVecG)));                    //         hue = rf - gf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[3], pAdd), _mm_and_ps(pMask[3], xmm_p4));                                  //         add = 4.0f;
    pH = _mm_or_ps(_mm_andnot_ps(pMask[0], pH), _mm_and_ps(pMask[0], _mm_div_ps(pH, pDelta)));                      //     hue /= delta; }}
}

// Converts HSV to RGB color space using SSE vectorization, processes 4 pixels simultaneously
inline void hsv_to_rgb(__m128 &pVecR, __m128 &pVecG, __m128 &pVecB, __m128 &pH, __m128 &pS, __m128 &pV, __m128 &pAdd)
{
    __m128 pMask[4], pIntH, pA;
    __m128i pxIntH;
    // Convert HSV to RGB 
    pIntH = _mm_floor_ps(pH);                                                                                       // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm_cvtps_epi32(pIntH);                                                                                // Convert to epi32
    pH = _mm_sub_ps(pH, pIntH);                                                                                     // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm_mul_ps(pV, pS);                                                                                        // Rpp32f vsat = v * sat;
    pAdd = _mm_mul_ps(pS, pH);                                                                                      // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm_sub_ps(pV, pS);                                                                                        // Rpp32f p = v - vsat;
    pH = _mm_sub_ps(pV, pAdd);                                                                                      // Rpp32f q = v - vsatf;
    pS = _mm_add_ps(pA, pAdd);                                                                                      // Rpp32f t = v - vsat + vsatf;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px0));                                                  // switch (hueIntegerPart) {case 0:
    pVecR = _mm_and_ps(pMask[0], pV);                                                                               //     rf = v;
    pVecG = _mm_and_ps(pMask[0], pS);                                                                               //     gf = t;
    pVecB = _mm_and_ps(pMask[0], pA);                                                                               //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px1));                                                  // case 1:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pH));                                    //     rf = q;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pA));                                    //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px2));                                                  // case 2:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pS));                                    //     bf = t; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px3));                                                  // case 3:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pH));                                    //     gf = q;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px4));                                                  // case 4:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pS));                                    //     rf = t;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px5));                                                  // case 5:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pV));                                    //     rf = v;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pH));                                    //     bf = q; break;}
}

#endif

// Converts RGB to HSV color space for a single pixel (scalar version)
inline void rgb_to_hsv(Rpp32f &rf, Rpp32f &gf, Rpp32f &bf, Rpp32f &hue, Rpp32f &sat, Rpp32f &val, Rpp32f &add)
{
    // Find maximum and minimum values among RGB components
    Rpp32f cmax = RPPMAX3(rf, gf, bf);
    Rpp32f cmin = RPPMIN3(rf, gf, bf);
    Rpp32f delta = cmax - cmin;

    // Initialize HSV values
    hue = 0.0f;
    sat = 0.0f;
    val = cmax;

    // Calculate saturation and hue if delta is not zero and max value is not zero
    if ((delta != 0) && (cmax != 0)) {
        sat = delta / cmax;
        // Calculate hue based on which RGB component is maximum
        if (cmax == rf)
        {
            hue = (gf - bf) / delta;
            add = 0.0f;
        }
        else if (cmax == gf)
        {
            sat = delta / cmax;
            hue = (bf - rf) / delta;
            add = 2.0f;
        } else {
            sat = delta / cmax;
            hue = (rf - gf) / delta;
            add = 4.0f;
        }
    }
}

// Converts HSV to RGB color space for a single pixel (scalar version)
inline void hsv_to_rgb(Rpp32f &rf, Rpp32f &gf, Rpp32f &bf, Rpp32f &hue, Rpp32f &sat, Rpp32f &val, Rpp32f &add)
{
    // Calculate intermediate values for RGB conversion
    Rpp32s hueIntegerPart = (Rpp32s)hue;
    Rpp32f f = hue - hueIntegerPart;
    Rpp32f p = val * (1.0f - sat);
    Rpp32f q = val * (1.0f - sat * f);
    Rpp32f t = val * (1.0f - sat * (1.0f - f));

    // Assign RGB values based on hue section (0-5)
    switch (hueIntegerPart) {
        case 0: rf = val; gf= t; bf = p; break;
        case 1: rf = q; gf = val; bf = p; break;
        case 2: rf = p; gf = val; bf = t; break;
        case 3: rf = p; gf = q; bf = val; break;
        case 4: rf = t; gf = p; bf = val; break;
        case 5: rf = val; gf = p; bf = q; break;
    }
}

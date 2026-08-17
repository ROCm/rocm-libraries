/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

inline void compute_tone_map_48_host(__m256* p) {
    __m256 pLumCoeffR = _mm256_set1_ps(0.2126f);
    __m256 pLumCoeffG = _mm256_set1_ps(0.7152f);
    __m256 pLumCoeffB = _mm256_set1_ps(0.0722f);
    __m256 pOne = _mm256_set1_ps(1.0f);

    // First 8 pixels: R=p[0], G=p[2], B=p[4]
    __m256 lum1 = _mm256_fmadd_ps(
        p[0], pLumCoeffR, _mm256_fmadd_ps(p[2], pLumCoeffG, _mm256_mul_ps(p[4], pLumCoeffB)));
    __m256 scale1 = _mm256_div_ps(pOne, _mm256_add_ps(pOne, lum1));
    p[0] = _mm256_mul_ps(p[0], scale1);
    p[2] = _mm256_mul_ps(p[2], scale1);
    p[4] = _mm256_mul_ps(p[4], scale1);

    // Second 8 pixels: R=p[1], G=p[3], B=p[5]
    __m256 lum2 = _mm256_fmadd_ps(
        p[1], pLumCoeffR, _mm256_fmadd_ps(p[3], pLumCoeffG, _mm256_mul_ps(p[5], pLumCoeffB)));
    __m256 scale2 = _mm256_div_ps(pOne, _mm256_add_ps(pOne, lum2));
    p[1] = _mm256_mul_ps(p[1], scale2);
    p[3] = _mm256_mul_ps(p[3], scale2);
    p[5] = _mm256_mul_ps(p[5], scale2);
}

inline void compute_tone_map_24_host(__m256* p) {
    __m256 pLumCoeffR = _mm256_set1_ps(0.2126f);
    __m256 pLumCoeffG = _mm256_set1_ps(0.7152f);
    __m256 pLumCoeffB = _mm256_set1_ps(0.0722f);
    __m256 pOne = _mm256_set1_ps(1.0f);

    // 8 pixels: R=p[0], G=p[1], B=p[2]
    __m256 luminance = _mm256_fmadd_ps(
        p[0], pLumCoeffR, _mm256_fmadd_ps(p[1], pLumCoeffG, _mm256_mul_ps(p[2], pLumCoeffB)));
    __m256 scale = _mm256_div_ps(pOne, _mm256_add_ps(pOne, luminance));
    p[0] = _mm256_mul_ps(p[0], scale);
    p[1] = _mm256_mul_ps(p[1], scale);
    p[2] = _mm256_mul_ps(p[2], scale);
}

inline void compute_tone_map_48_u8_host(__m256* p) {
    __m256 pLumCoeffR = _mm256_set1_ps(0.2126f);
    __m256 pLumCoeffG = _mm256_set1_ps(0.7152f);
    __m256 pLumCoeffB = _mm256_set1_ps(0.0722f);
    __m256 pOne = _mm256_set1_ps(1.0f);
    __m256 p1o255 = _mm256_set1_ps(1.0f / 255.0f);
    __m256 p255 = _mm256_set1_ps(255.0f);

    // Normalize from [0,255] to [0,1]
    __m256 r1 = _mm256_mul_ps(p[0], p1o255);
    __m256 g1 = _mm256_mul_ps(p[2], p1o255);
    __m256 b1 = _mm256_mul_ps(p[4], p1o255);
    __m256 r2 = _mm256_mul_ps(p[1], p1o255);
    __m256 g2 = _mm256_mul_ps(p[3], p1o255);
    __m256 b2 = _mm256_mul_ps(p[5], p1o255);

    // First 8 pixels
    __m256 lum1 = _mm256_fmadd_ps(r1, pLumCoeffR,
                                  _mm256_fmadd_ps(g1, pLumCoeffG, _mm256_mul_ps(b1, pLumCoeffB)));
    __m256 scale1 = _mm256_div_ps(pOne, _mm256_add_ps(pOne, lum1));
    p[0] = _mm256_mul_ps(_mm256_mul_ps(r1, scale1), p255);
    p[2] = _mm256_mul_ps(_mm256_mul_ps(g1, scale1), p255);
    p[4] = _mm256_mul_ps(_mm256_mul_ps(b1, scale1), p255);

    // Second 8 pixels
    __m256 lum2 = _mm256_fmadd_ps(r2, pLumCoeffR,
                                  _mm256_fmadd_ps(g2, pLumCoeffG, _mm256_mul_ps(b2, pLumCoeffB)));
    __m256 scale2 = _mm256_div_ps(pOne, _mm256_add_ps(pOne, lum2));
    p[1] = _mm256_mul_ps(_mm256_mul_ps(r2, scale2), p255);
    p[3] = _mm256_mul_ps(_mm256_mul_ps(g2, scale2), p255);
    p[5] = _mm256_mul_ps(_mm256_mul_ps(b2, scale2), p255);
}

// Single-channel (greyscale) SIMD helpers: L = pixel, out = pixel / (1 + pixel)
inline void compute_tone_map_16_u8_host(__m256& p0, __m256& p1) {
    __m256 pOne = _mm256_set1_ps(1.0f);
    __m256 p1o255 = _mm256_set1_ps(1.0f / 255.0f);
    __m256 p255 = _mm256_set1_ps(255.0f);
    __m256 v0 = _mm256_mul_ps(p0, p1o255);
    __m256 v1 = _mm256_mul_ps(p1, p1o255);
    p0 = _mm256_mul_ps(_mm256_div_ps(v0, _mm256_add_ps(pOne, v0)), p255);
    p1 = _mm256_mul_ps(_mm256_div_ps(v1, _mm256_add_ps(pOne, v1)), p255);
}

inline void compute_tone_map_8_host(__m256& p) {
    __m256 pOne = _mm256_set1_ps(1.0f);
    p = _mm256_div_ps(p, _mm256_add_ps(pOne, p));
}

inline void compute_tone_map_scalar_host(Rpp32f& R, Rpp32f& G, Rpp32f& B, Rpp32f invGamma) {
    Rpp32f L = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    Rpp32f scale = 1.0f / (1.0f + L);
    R *= scale;
    G *= scale;
    B *= scale;
    if (invGamma != 1.0f) {
        R = std::pow(R, invGamma);
        G = std::pow(G, invGamma);
        B = std::pow(B, invGamma);
    }
}

RppStatus tone_map_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* gammaTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];
        Rpp32f invGamma = (gamma > 0) ? (1.0f / gamma) : 1.0f;

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

        Rpp32u useGamma = (invGamma != 1.0f) ? 1 : 0;

        // Tone map with fused output-layout toggle (NHWC -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);                // simd loads
                        compute_tone_map_48_u8_host(p);  // tone mapping
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)(srcPtrTemp[0]) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(srcPtrTemp[1]) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(R * 255.0f));
                    *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(G * 255.0f));
                    *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(B * 255.0f));

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

        // Tone map with fused output-layout toggle (NCHW -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);   // simd loads
                        compute_tone_map_48_u8_host(p);  // tone mapping
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)(*srcPtrTempR) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(*srcPtrTempG) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(*srcPtrTempB) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(R * 255.0f));
                    dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(G * 255.0f));
                    dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(B * 255.0f));

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

        // Tone map without fused output-layout toggle (NHWC -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);                // simd loads
                        compute_tone_map_48_u8_host(p);  // tone mapping
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)(srcPtrTemp[0]) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(srcPtrTemp[1]) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(R * 255.0f));
                    dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(G * 255.0f));
                    dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(B * 255.0f));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Tone map without fused output-layout toggle (NCHW -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);   // simd loads
                        compute_tone_map_48_u8_host(p);  // tone mapping
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)(*srcPtrTempR) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(*srcPtrTempG) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(*srcPtrTempB) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR++ = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(R * 255.0f));
                    *dstPtrTempG++ = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(G * 255.0f));
                    *dstPtrTempB++ = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(B * 255.0f));

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
            }
        }

        // Tone map without fused output-layout toggle single channel (PLN1 -> PLN1)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == dstDescPtr->layout)) {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);
                        compute_tone_map_16_u8_host(p[0], p[1]);
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);

                        srcPtrTemp += 16;
                        dstPtrTemp += 16;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f val = (Rpp32f)(*srcPtrTemp) * ONE_OVER_255;
                    val = val / (1.0f + val);
                    if (invGamma != 1.0f) val = std::pow(val, invGamma);
                    *dstPtrTemp = (Rpp8u)RPPPIXELCHECK(std::nearbyintf(val * 255.0f));

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

RppStatus tone_map_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* gammaTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];
        Rpp32f invGamma = (gamma > 0) ? (1.0f / gamma) : 1.0f;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32u useGamma = (invGamma != 1.0f) ? 1 : 0;

        // Tone map with fused output-layout toggle (NHWC -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);             // simd loads
                        compute_tone_map_24_host(p);  // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = srcPtrTemp[0];
                    Rpp32f G = srcPtrTemp[1];
                    Rpp32f B = srcPtrTemp[2];
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR = RPPPIXELCHECKF32(R);
                    *dstPtrTempG = RPPPIXELCHECKF32(G);
                    *dstPtrTempB = RPPPIXELCHECKF32(B);

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

        // Tone map with fused output-layout toggle (NCHW -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_24_host(p);    // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = *srcPtrTempR;
                    Rpp32f G = *srcPtrTempG;
                    Rpp32f B = *srcPtrTempB;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(B);

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

        // Tone map without fused output-layout toggle (NHWC -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);             // simd loads
                        compute_tone_map_24_host(p);  // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = srcPtrTemp[0];
                    Rpp32f G = srcPtrTemp[1];
                    Rpp32f B = srcPtrTemp[2];
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(B);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Tone map without fused output-layout toggle (NCHW -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_24_host(p);    // tone mapping
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
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = *srcPtrTempR;
                    Rpp32f G = *srcPtrTempG;
                    Rpp32f B = *srcPtrTempB;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR++ = RPPPIXELCHECKF32(R);
                    *dstPtrTempG++ = RPPPIXELCHECKF32(G);
                    *dstPtrTempB++ = RPPPIXELCHECKF32(B);

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
            }
        }

        // Tone map without fused output-layout toggle single channel (PLN1 -> PLN1)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == dstDescPtr->layout)) {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        __m256 p;
                        p = _mm256_loadu_ps(srcPtrTemp);
                        compute_tone_map_8_host(p);
                        p = _mm256_max_ps(_mm256_setzero_ps(),
                                          _mm256_min_ps(p, _mm256_set1_ps(1.0f)));
                        _mm256_storeu_ps(dstPtrTemp, p);

                        srcPtrTemp += 8;
                        dstPtrTemp += 8;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f val = *srcPtrTemp;
                    val = val / (1.0f + val);
                    if (invGamma != 1.0f) val = std::pow(val, invGamma);
                    *dstPtrTemp = RPPPIXELCHECKF32(val);

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

RppStatus tone_map_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* gammaTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];
        Rpp32f invGamma = (gamma > 0) ? (1.0f / gamma) : 1.0f;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32u useGamma = (invGamma != 1.0f) ? 1 : 0;

        // Tone map with fused output-layout toggle (NHWC -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[3];

                        rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);             // simd loads
                        compute_tone_map_24_host(p);  // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)srcPtrTemp[0];
                    Rpp32f G = (Rpp32f)srcPtrTemp[1];
                    Rpp32f B = (Rpp32f)srcPtrTemp[2];
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR = (Rpp16f)RPPPIXELCHECKF32(R);
                    *dstPtrTempG = (Rpp16f)RPPPIXELCHECKF32(G);
                    *dstPtrTempB = (Rpp16f)RPPPIXELCHECKF32(B);

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

        // Tone map with fused output-layout toggle (NCHW -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_24_host(p);    // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)(*srcPtrTempR);
                    Rpp32f G = (Rpp32f)(*srcPtrTempG);
                    Rpp32f B = (Rpp32f)(*srcPtrTempB);
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp16f)RPPPIXELCHECKF32(R);
                    dstPtrTemp[1] = (Rpp16f)RPPPIXELCHECKF32(G);
                    dstPtrTemp[2] = (Rpp16f)RPPPIXELCHECKF32(B);

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

        // Tone map without fused output-layout toggle (NHWC -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);             // simd loads
                        compute_tone_map_24_host(p);  // tone mapping
                        rpp_pixel_check_0to1(p, 3);
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)srcPtrTemp[0];
                    Rpp32f G = (Rpp32f)srcPtrTemp[1];
                    Rpp32f B = (Rpp32f)srcPtrTemp[2];
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp16f)RPPPIXELCHECKF32(R);
                    dstPtrTemp[1] = (Rpp16f)RPPPIXELCHECKF32(G);
                    dstPtrTemp[2] = (Rpp16f)RPPPIXELCHECKF32(B);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Tone map without fused output-layout toggle (NCHW -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_24_host(p);    // tone mapping
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
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)(*srcPtrTempR);
                    Rpp32f G = (Rpp32f)(*srcPtrTempG);
                    Rpp32f B = (Rpp32f)(*srcPtrTempB);
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR++ = (Rpp16f)RPPPIXELCHECKF32(R);
                    *dstPtrTempG++ = (Rpp16f)RPPPIXELCHECKF32(G);
                    *dstPtrTempB++ = (Rpp16f)RPPPIXELCHECKF32(B);

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
            }
        }

        // Tone map without fused output-layout toggle single channel (PLN1 -> PLN1)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == dstDescPtr->layout)) {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        __m256 p;
                        p = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)srcPtrTemp));
                        compute_tone_map_8_host(p);
                        p = _mm256_max_ps(_mm256_setzero_ps(),
                                          _mm256_min_ps(p, _mm256_set1_ps(1.0f)));
                        _mm_storeu_si128(
                            (__m128i*)dstPtrTemp,
                            _mm256_cvtps_ph(p, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                        srcPtrTemp += 8;
                        dstPtrTemp += 8;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f val = (Rpp32f)(*srcPtrTemp);
                    val = val / (1.0f + val);
                    if (invGamma != 1.0f) val = std::pow(val, invGamma);
                    *dstPtrTemp = (Rpp16f)RPPPIXELCHECKF32(val);

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

RppStatus tone_map_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* gammaTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];
        Rpp32f invGamma = (gamma > 0) ? (1.0f / gamma) : 1.0f;

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

        Rpp32u useGamma = (invGamma != 1.0f) ? 1 : 0;

        // Tone map with fused output-layout toggle (NHWC -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);  // simd loads
                        compute_tone_map_48_u8_host(
                            p);  // tone mapping (values are in [0,255] after i8 load)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)(srcPtrTemp[0] + 128) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(srcPtrTemp[1] + 128) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(srcPtrTemp[2] + 128) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(R * 255.0f) - 128);
                    *dstPtrTempG = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(G * 255.0f) - 128);
                    *dstPtrTempB = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(B * 255.0f) - 128);

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

        // Tone map with fused output-layout toggle (NCHW -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_48_u8_host(
                            p);  // tone mapping (values are in [0,255] after i8 load)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)((*srcPtrTempR) + 128) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)((*srcPtrTempG) + 128) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)((*srcPtrTempB) + 128) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(R * 255.0f) - 128);
                    dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(G * 255.0f) - 128);
                    dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(B * 255.0f) - 128);

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

        // Tone map without fused output-layout toggle (NHWC -> NHWC)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp,
                                      p);  // simd loads
                        compute_tone_map_48_u8_host(
                            p);  // tone mapping (values are in [0,255] after i8 load)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                       p);  // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    Rpp32f R = (Rpp32f)(srcPtrTemp[0] + 128) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)(srcPtrTemp[1] + 128) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)(srcPtrTemp[2] + 128) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(R * 255.0f) - 128);
                    dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(G * 255.0f) - 128);
                    dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(B * 255.0f) - 128);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Tone map without fused output-layout toggle (NCHW -> NCHW)
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
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                      srcPtrTempB, p);  // simd loads
                        compute_tone_map_48_u8_host(
                            p);  // tone mapping (values are in [0,255] after i8 load)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                       dstPtrTempB, p);  // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f R = (Rpp32f)((*srcPtrTempR) + 128) * ONE_OVER_255;
                    Rpp32f G = (Rpp32f)((*srcPtrTempG) + 128) * ONE_OVER_255;
                    Rpp32f B = (Rpp32f)((*srcPtrTempB) + 128) * ONE_OVER_255;
                    compute_tone_map_scalar_host(R, G, B, invGamma);
                    *dstPtrTempR++ = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(R * 255.0f) - 128);
                    *dstPtrTempG++ = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(G * 255.0f) - 128);
                    *dstPtrTempB++ = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(B * 255.0f) - 128);

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
            }
        }

        // Tone map without fused output-layout toggle single channel (PLN1 -> PLN1)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == dstDescPtr->layout)) {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                if (!useGamma) {
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);
                        compute_tone_map_16_u8_host(p[0], p[1]);
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);

                        srcPtrTemp += 16;
                        dstPtrTemp += 16;
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f val = (Rpp32f)((*srcPtrTemp) + 128) * ONE_OVER_255;
                    val = val / (1.0f + val);
                    if (invGamma != 1.0f) val = std::pow(val, invGamma);
                    *dstPtrTemp = (Rpp8s)RPPPIXELCHECKI8(std::nearbyintf(val * 255.0f) - 128);

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

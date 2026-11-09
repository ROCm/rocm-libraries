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
#include "rpp_cpu_simd_math.hpp"

inline void compute_vignette_48_host(__m256* p, __m256& pMultiplier, __m256& pILocComponent,
                                     __m256& pJLocComponent) {
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);  // vignette adjustment
    p[2] = _mm256_mul_ps(p[2], pGaussianValue);  // vignette adjustment
    p[4] = _mm256_mul_ps(p[4], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);  // vignette adjustment
    p[3] = _mm256_mul_ps(p[3], pGaussianValue);  // vignette adjustment
    p[5] = _mm256_mul_ps(p[5], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_24_host(__m256* p, __m256& pMultiplier, __m256& pILocComponent,
                                     __m256& pJLocComponent) {
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);  // vignette adjustment
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);  // vignette adjustment
    p[2] = _mm256_mul_ps(p[2], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_16_host(__m256* p, __m256& pMultiplier, __m256& pILocComponent,
                                     __m256& pJLocComponent) {
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_8_host(__m256* p, __m256& pMultiplier, __m256& pILocComponent,
                                    __m256& pJLocComponent) {
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(
        _mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);  // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void vignette_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8u* dstPtrImage,
                                     RpptDescPtr dstDescPtr, RpptROI roi, Rpp32f multiplier,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
    Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);

    Rpp32u bufferLength = roi.xywhROI.roiWidth;
    Rpp32u alignedLength = bufferLength & ~15;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    __m256 pMultiplier = _mm256_set1_ps(multiplier);
    __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

    // Vignette with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_48_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTemp += 48;
                dstPtrTempR += 16;
                dstPtrTempG += 16;
                dstPtrTempB += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[0] * gaussianValue));
                *dstPtrTempG++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[1] * gaussianValue));
                *dstPtrTempB++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[2] * gaussianValue));

                srcPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 16) ? (alignedLength - 16) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 16) {
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_48_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
                srcPtrTempR += 16;
                srcPtrTempG += 16;
                srcPtrTempB += 16;
                dstPtrTemp += 48;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempR * gaussianValue));
                dstPtrTemp[1] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempG * gaussianValue));
                dstPtrTemp[2] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempB * gaussianValue));

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 16) ? (alignedLength - 16) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 16) {
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_48_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
                srcPtrTemp += 48;
                dstPtrTemp += 48;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[0] * gaussianValue));
                dstPtrTemp[1] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[1] * gaussianValue));
                dstPtrTemp[2] =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)srcPtrTemp[2] * gaussianValue));

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NCHW)
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
            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_48_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTempR += 16;
                srcPtrTempG += 16;
                srcPtrTempB += 16;
                dstPtrTempR += 16;
                dstPtrTempG += 16;
                dstPtrTempB += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempR * gaussianValue));
                *dstPtrTempG++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempG * gaussianValue));
                *dstPtrTempB++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTempB * gaussianValue));

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
    }

    // Vignette without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[2];
                rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);
                compute_vignette_16_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);
                srcPtrTemp += 16;
                dstPtrTemp += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));
                *dstPtrTemp++ =
                    (Rpp8u)RPPPIXELCHECK(nearbyintf((Rpp32f)*srcPtrTemp * gaussianValue));
                srcPtrTemp++;
            }
        }
    }
}

RppStatus vignette_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* vignetteIntensityTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensity = vignetteIntensityTensor[batchCount];
        Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);
        Rpp32f radius = std::max(halfHeight, halfWidth);
        Rpp32f multiplier = -(0.25f * intensity) / (radius * radius);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        vignette_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, multiplier,
                                 layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void vignette_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr, RpptROI roi,
                                       Rpp32f multiplier, RppLayoutParams layoutParams,
                                       Rpp32u intraThreads) {
    Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
    Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);

    Rpp32u bufferLength = roi.xywhROI.roiWidth;
    Rpp32u alignedLength = bufferLength & ~7;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    __m256 pMultiplier = _mm256_set1_ps(multiplier);
    __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

    // Vignette with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTemp += 24;
                dstPtrTempR += 8;
                dstPtrTempG += 8;
                dstPtrTempB += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = RPPPIXELCHECKF32(srcPtrTemp[0] * gaussianValue);
                *dstPtrTempG++ = RPPPIXELCHECKF32(srcPtrTemp[1] * gaussianValue);
                *dstPtrTempB++ = RPPPIXELCHECKF32(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);
                srcPtrTempR += 8;
                srcPtrTempG += 8;
                srcPtrTempB += 8;
                dstPtrTemp += 24;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = RPPPIXELCHECKF32(*srcPtrTempR * gaussianValue);
                dstPtrTemp[1] = RPPPIXELCHECKF32(*srcPtrTempG * gaussianValue);
                dstPtrTemp[2] = RPPPIXELCHECKF32(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);
                srcPtrTemp += 24;
                dstPtrTemp += 24;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = RPPPIXELCHECKF32(srcPtrTemp[0] * gaussianValue);
                dstPtrTemp[1] = RPPPIXELCHECKF32(srcPtrTemp[1] * gaussianValue);
                dstPtrTemp[2] = RPPPIXELCHECKF32(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NCHW)
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
            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[6];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTempR += 8;
                srcPtrTempG += 8;
                srcPtrTempB += 8;
                dstPtrTempR += 8;
                dstPtrTempG += 8;
                dstPtrTempB += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = RPPPIXELCHECKF32(*srcPtrTempR * gaussianValue);
                *dstPtrTempG++ = RPPPIXELCHECKF32(*srcPtrTempG * gaussianValue);
                *dstPtrTempB++ = RPPPIXELCHECKF32(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
    }

    // Vignette without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[2];
                rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);
                compute_vignette_8_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);
                srcPtrTemp += 8;
                dstPtrTemp += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));
                *dstPtrTemp++ = RPPPIXELCHECKF32(*srcPtrTemp * gaussianValue);
                srcPtrTemp++;
            }
        }
    }
}

RppStatus vignette_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* vignetteIntensityTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensity = vignetteIntensityTensor[batchCount];
        Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);
        Rpp32f radius = std::max(halfHeight, halfWidth);
        Rpp32f multiplier = -(0.25f * intensity) / (radius * radius);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        vignette_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi,
                                   multiplier, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void vignette_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8s* dstPtrImage,
                                     RpptDescPtr dstDescPtr, RpptROI roi, Rpp32f multiplier,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
    Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);

    Rpp32u bufferLength = roi.xywhROI.roiWidth;
    Rpp32u alignedLength = bufferLength & ~15;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    __m256 pMultiplier = _mm256_set1_ps(multiplier);
    __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

    // Vignette with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[3];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTemp += 48;
                dstPtrTempR += 16;
                dstPtrTempG += 16;
                dstPtrTempB += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = RPPPIXELCHECKI8(srcPtrTemp[0] * gaussianValue);
                *dstPtrTempG++ = RPPPIXELCHECKI8(srcPtrTemp[1] * gaussianValue);
                *dstPtrTempB++ = RPPPIXELCHECKI8(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 16) ? (alignedLength - 16) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 16) {
                __m256 p[3];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);
                srcPtrTempR += 16;
                srcPtrTempG += 16;
                srcPtrTempB += 16;
                dstPtrTemp += 48;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = RPPPIXELCHECKI8(*srcPtrTempR * gaussianValue);
                dstPtrTemp[1] = RPPPIXELCHECKI8(*srcPtrTempG * gaussianValue);
                dstPtrTemp[2] = RPPPIXELCHECKI8(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 16) ? (alignedLength - 16) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 16) {
                __m256 p[3];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);
                srcPtrTemp += 48;
                dstPtrTemp += 48;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = RPPPIXELCHECKI8(srcPtrTemp[0] * gaussianValue);
                dstPtrTemp[1] = RPPPIXELCHECKI8(srcPtrTemp[1] * gaussianValue);
                dstPtrTemp[2] = RPPPIXELCHECKI8(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NCHW)
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
            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTempR += 16;
                srcPtrTempG += 16;
                srcPtrTempB += 16;
                dstPtrTempR += 16;
                dstPtrTempG += 16;
                dstPtrTempB += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = RPPPIXELCHECKI8(*srcPtrTempR * gaussianValue);
                *dstPtrTempG++ = RPPPIXELCHECKI8(*srcPtrTempG * gaussianValue);
                *dstPtrTempB++ = RPPPIXELCHECKI8(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
    }

    // Vignette without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                __m256 p[2];
                rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);
                compute_vignette_16_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);
                srcPtrTemp += 16;
                dstPtrTemp += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));
                *dstPtrTemp++ = RPPPIXELCHECKI8(*srcPtrTemp * gaussianValue);
                srcPtrTemp++;
            }
        }
    }
}

RppStatus vignette_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* vignetteIntensityTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensity = vignetteIntensityTensor[batchCount];
        Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);
        Rpp32f radius = std::max(halfHeight, halfWidth);
        Rpp32f multiplier = -(0.25f * intensity) / (radius * radius);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        vignette_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, multiplier,
                                 layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void vignette_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr, RpptROI roi,
                                       Rpp32f multiplier, RppLayoutParams layoutParams,
                                       Rpp32u intraThreads) {
    Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
    Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);

    Rpp32u bufferLength = roi.xywhROI.roiWidth;
    Rpp32u alignedLength = bufferLength & ~7;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    __m256 pMultiplier = _mm256_set1_ps(multiplier);
    __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

    // Vignette with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTemp += 24;
                dstPtrTempR += 8;
                dstPtrTempG += 8;
                dstPtrTempB += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = (Rpp16f)(srcPtrTemp[0] * gaussianValue);
                *dstPtrTempG++ = (Rpp16f)(srcPtrTemp[1] * gaussianValue);
                *dstPtrTempB++ = (Rpp16f)(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);
                srcPtrTempR += 8;
                srcPtrTempG += 8;
                srcPtrTempB += 8;
                dstPtrTemp += 24;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = (Rpp16f)(*srcPtrTempR * gaussianValue);
                dstPtrTemp[1] = (Rpp16f)(*srcPtrTempG * gaussianValue);
                dstPtrTemp[2] = (Rpp16f)(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

        Rpp32u alignedLengthPkd = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd; vectorLoopCount += 8) {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);
                srcPtrTemp += 24;
                dstPtrTemp += 24;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                dstPtrTemp[0] = (Rpp16f)(srcPtrTemp[0] * gaussianValue);
                dstPtrTemp[1] = (Rpp16f)(srcPtrTemp[1] * gaussianValue);
                dstPtrTemp[2] = (Rpp16f)(srcPtrTemp[2] * gaussianValue);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Vignette with fused output-layout toggle (NCHW -> NCHW)
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
            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                *dstPtrTempB;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTempR = dstPtrRowR + i * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + i * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[6];
                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);
                compute_vignette_24_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);
                srcPtrTempR += 8;
                srcPtrTempG += 8;
                srcPtrTempB += 8;
                dstPtrTempR += 8;
                dstPtrTempG += 8;
                dstPtrTempB += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));

                *dstPtrTempR++ = (Rpp16f)(*srcPtrTempR * gaussianValue);
                *dstPtrTempG++ = (Rpp16f)(*srcPtrTempG * gaussianValue);
                *dstPtrTempB++ = (Rpp16f)(*srcPtrTempB * gaussianValue);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
            }
        }
    }

    // Vignette without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            Rpp32s iLoc = i - halfHeight;
            Rpp32f iLocComponent = iLoc * iLoc;
            __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
            __m256 pJLocComponent = _mm256_sub_ps(avx_pDstLocInit, pHalfWidth);

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 p[2];
                rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, p);
                compute_vignette_8_host(p, pMultiplier, pILocComponent, pJLocComponent);
                rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);
                srcPtrTemp += 8;
                dstPtrTemp += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32s jLoc = vectorLoopCount - halfWidth;
                Rpp32f jLocComponent = jLoc * jLoc;
                Rpp32f gaussianValue =
                    std::exp((iLocComponent * multiplier)) * std::exp((jLocComponent * multiplier));
                *dstPtrTemp++ = (Rpp16f)(*srcPtrTemp * gaussianValue);
                srcPtrTemp++;
            }
        }
    }
}

RppStatus vignette_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* vignetteIntensityTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f intensity = vignetteIntensityTensor[batchCount];
        Rpp32s halfHeight = (Rpp32s)(roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s)(roi.xywhROI.roiWidth >> 1);
        Rpp32f radius = std::max(halfHeight, halfWidth);
        Rpp32f multiplier = -(0.25f * intensity) / (radius * radius);

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        vignette_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi,
                                   multiplier, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

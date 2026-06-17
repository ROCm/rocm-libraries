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
#include "rpp_cpu_simd_math.hpp"

constexpr Rpp32s HISTOGRAM_BINS = 256;

// Coefficients for RGB to YCbCr Conversion
const Rpp32f coeffYR = 0.299000f;
const Rpp32f coeffYG = 0.587000f;
const Rpp32f coeffYB = 0.114000f;
const Rpp32f coeffCbR = -0.168736f;
const Rpp32f coeffCbG = -0.331264f;
const Rpp32f coeffCbB = 0.500000f;
const Rpp32f coeffCrR = 0.500000f;
const Rpp32f coeffCrG = -0.418688f;
const Rpp32f coeffCrB = -0.081312f;

// Coefficients for YCbCr to RGB Conversion
const Rpp32f coeffRCr = 1.402000f;
const Rpp32f coeffGCb = 0.344136f;
const Rpp32f coeffGCr = 0.714136f;
const Rpp32f coeffBCb = 1.772000f;

#if __AVX2__
// AVX2 RGB to YCbCr coefficients
const __m256 pCoeffYR = _mm256_set1_ps(coeffYR);
const __m256 pCoeffYG = _mm256_set1_ps(coeffYG);
const __m256 pCoeffYB = _mm256_set1_ps(coeffYB);
const __m256 pCoeffCbR = _mm256_set1_ps(coeffCbR);
const __m256 pCoeffCbG = _mm256_set1_ps(coeffCbG);
const __m256 pCoeffCbB = _mm256_set1_ps(coeffCbB);
const __m256 pCoeffCrR = _mm256_set1_ps(coeffCrR);
const __m256 pCoeffCrG = _mm256_set1_ps(coeffCrG);
const __m256 pCoeffCrB = _mm256_set1_ps(coeffCrB);

// AVX2 YCbCr to RGB coefficients
const __m256 pCoeffRCr = _mm256_set1_ps(coeffRCr);
const __m256 pCoeffGCb = _mm256_set1_ps(coeffGCb);
const __m256 pCoeffGCr = _mm256_set1_ps(coeffGCr);
const __m256 pCoeffBCb = _mm256_set1_ps(coeffBCb);
#endif

// Scalar RGB to YCbCr conversion
inline void rgb_to_ycbcr_compute(Rpp8u* srcR, Rpp8u* srcG, Rpp8u* srcB, Rpp8u* dstY, Rpp8u* dstCb,
                                 Rpp8u* dstCr) {
    Rpp32f r = static_cast<Rpp32f>(*srcR);
    Rpp32f g = static_cast<Rpp32f>(*srcG);
    Rpp32f b = static_cast<Rpp32f>(*srcB);

    Rpp32f yVal = fmaf(r, coeffYR, fmaf(g, coeffYG, b * coeffYB));
    Rpp32f cbVal = fmaf(r, coeffCbR, fmaf(g, coeffCbG, fmaf(b, coeffCbB, 128.0f)));
    Rpp32f crVal = fmaf(r, coeffCrR, fmaf(g, coeffCrG, fmaf(b, coeffCrB, 128.0f)));

    *dstY = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(yVal)));
    *dstCb = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(cbVal)));
    *dstCr = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(crVal)));
}

// Scalar YCbCr to RGB conversion
inline void ycbcr_to_rgb_compute(Rpp8u* srcY, Rpp8u* srcCb, Rpp8u* srcCr, Rpp8u* dstR, Rpp8u* dstG,
                                 Rpp8u* dstB) {
    Rpp32f yVal = static_cast<Rpp32f>(*srcY);
    Rpp32f cbVal = static_cast<Rpp32f>(*srcCb) - 128.0f;
    Rpp32f crVal = static_cast<Rpp32f>(*srcCr) - 128.0f;

    Rpp32f r = yVal + coeffRCr * crVal;
    Rpp32f g = yVal - coeffGCb * cbVal - coeffGCr * crVal;
    Rpp32f b = yVal + coeffBCb * cbVal;

    *dstR = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(r)));
    *dstG = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(g)));
    *dstB = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(b)));
}

#if __AVX2__
// AVX2 RGB to YCbCr conversion for 16 pixels
inline void rgb_to_ycbcr_avx(__m256* p) {
    __m256 y0 = _mm256_fmadd_ps(p[0], pCoeffYR,
                                _mm256_fmadd_ps(p[2], pCoeffYG, _mm256_mul_ps(p[4], pCoeffYB)));
    __m256 cb0 = _mm256_fmadd_ps(
        p[0], pCoeffCbR,
        _mm256_fmadd_ps(p[2], pCoeffCbG, _mm256_fmadd_ps(p[4], pCoeffCbB, avx_p128)));
    __m256 cr0 = _mm256_fmadd_ps(
        p[0], pCoeffCrR,
        _mm256_fmadd_ps(p[2], pCoeffCrG, _mm256_fmadd_ps(p[4], pCoeffCrB, avx_p128)));

    p[0] = _mm256_min_ps(_mm256_max_ps(y0, avx_p0), avx_p255);
    p[2] = _mm256_min_ps(_mm256_max_ps(cb0, avx_p0), avx_p255);
    p[4] = _mm256_min_ps(_mm256_max_ps(cr0, avx_p0), avx_p255);

    __m256 y1 = _mm256_fmadd_ps(p[1], pCoeffYR,
                                _mm256_fmadd_ps(p[3], pCoeffYG, _mm256_mul_ps(p[5], pCoeffYB)));
    __m256 cb1 = _mm256_fmadd_ps(
        p[1], pCoeffCbR,
        _mm256_fmadd_ps(p[3], pCoeffCbG, _mm256_fmadd_ps(p[5], pCoeffCbB, avx_p128)));
    __m256 cr1 = _mm256_fmadd_ps(
        p[1], pCoeffCrR,
        _mm256_fmadd_ps(p[3], pCoeffCrG, _mm256_fmadd_ps(p[5], pCoeffCrB, avx_p128)));

    p[1] = _mm256_min_ps(_mm256_max_ps(y1, avx_p0), avx_p255);
    p[3] = _mm256_min_ps(_mm256_max_ps(cb1, avx_p0), avx_p255);
    p[5] = _mm256_min_ps(_mm256_max_ps(cr1, avx_p0), avx_p255);
}

// AVX2 YCbCr to RGB conversion for 16 pixels
inline void ycbcr_to_rgb_avx(__m256* p) {
    __m256 cb0 = _mm256_sub_ps(p[2], avx_p128);
    __m256 cr0 = _mm256_sub_ps(p[4], avx_p128);

    __m256 r0 = _mm256_fmadd_ps(cr0, pCoeffRCr, p[0]);
    __m256 g0 = _mm256_fnmadd_ps(cr0, pCoeffGCr, _mm256_fnmadd_ps(cb0, pCoeffGCb, p[0]));
    __m256 b0 = _mm256_fmadd_ps(cb0, pCoeffBCb, p[0]);

    p[0] = _mm256_min_ps(_mm256_max_ps(r0, avx_p0), avx_p255);
    p[2] = _mm256_min_ps(_mm256_max_ps(g0, avx_p0), avx_p255);
    p[4] = _mm256_min_ps(_mm256_max_ps(b0, avx_p0), avx_p255);

    __m256 cb1 = _mm256_sub_ps(p[3], avx_p128);
    __m256 cr1 = _mm256_sub_ps(p[5], avx_p128);

    __m256 r1 = _mm256_fmadd_ps(cr1, pCoeffRCr, p[1]);
    __m256 g1 = _mm256_fnmadd_ps(cr1, pCoeffGCr, _mm256_fnmadd_ps(cb1, pCoeffGCb, p[1]));
    __m256 b1 = _mm256_fmadd_ps(cb1, pCoeffBCb, p[1]);

    p[1] = _mm256_min_ps(_mm256_max_ps(r1, avx_p0), avx_p255);
    p[3] = _mm256_min_ps(_mm256_max_ps(g1, avx_p0), avx_p255);
    p[5] = _mm256_min_ps(_mm256_max_ps(b1, avx_p0), avx_p255);
}
#endif

// Collect histogram from planar image buffer
inline void collect_hist_pln_tensor_host(Rpp8u* srcPtr, Rpp32u* hist, Rpp32u roiWidth,
                                         Rpp32u roiHeight, Rpp32u rowStride) {
    for (int y = 0; y < roiHeight; y++) {
        Rpp8u* srcRow = srcPtr + y * rowStride;
        for (int x = 0; x < roiWidth; x++) hist[srcRow[x]]++;
    }
}

// Collect histogram from Y channel buffer
inline void collect_hist_y_buffer(const Rpp8u* yBuf, Rpp32u* hist, Rpp32u pixels) {
    for (Rpp32u i = 0; i < pixels; i++) hist[yBuf[i]]++;
}

// Build equalization LUT from histogram
inline void build_lut_from_hist_host(const Rpp32u* hist, Rpp8u* lut, Rpp32u imgSize) {
    Rpp32u cdf[HISTOGRAM_BINS];
    Rpp32u cdfAccum = 0;
    Rpp32u minCdf = 0;

    for (int i = 0; i < HISTOGRAM_BINS; i++) {
        cdfAccum += hist[i];
        cdf[i] = cdfAccum;
        // Find the first non-zero CDF value (minimum CDF) required for histogram equalization
        // formula: equalized_value = ((cdf[i] - minCdf) / (imgSize - minCdf)) * 255
        if (!minCdf && cdf[i]) minCdf = cdf[i];
    }

    Rpp32f denominator = std::max(static_cast<Rpp32f>(imgSize - minCdf), 1.0f);
    bool isUniform = (minCdf == imgSize);

    if (isUniform) {
        for (int i = 0; i < HISTOGRAM_BINS; i++) lut[i] = static_cast<Rpp8u>(i);
        return;
    }

    const Rpp32f multScalar = 255.0f / denominator;
    int vectorLoopCount = 0;

#if __AVX2__
    __m256 pMinCdf = _mm256_set1_ps(static_cast<Rpp32f>(minCdf));
    __m256 pMult = _mm256_set1_ps(multScalar);
    for (; vectorLoopCount <= HISTOGRAM_BINS - 16; vectorLoopCount += 16) {
        __m256i ci0 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(cdf + vectorLoopCount));
        __m256 cf0 = _mm256_cvtepi32_ps(ci0);
        __m256 r0 = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(cf0, pMinCdf), pMult), avx_p255);
        __m256i ri0 = _mm256_cvtps_epi32(r0);

        __m256i ci1 =
            _mm256_loadu_si256(reinterpret_cast<__m256i const*>(cdf + vectorLoopCount + 8));
        __m256 cf1 = _mm256_cvtepi32_ps(ci1);
        __m256 r1 = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(cf1, pMinCdf), pMult), avx_p255);
        __m256i ri1 = _mm256_cvtps_epi32(r1);

        __m128i pack16_0 =
            _mm_packs_epi32(_mm256_castsi256_si128(ri0), _mm256_extracti128_si256(ri0, 1));
        __m128i pack16_1 =
            _mm_packs_epi32(_mm256_castsi256_si128(ri1), _mm256_extracti128_si256(ri1, 1));
        __m128i pack8 = _mm_packus_epi16(pack16_0, pack16_1);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(lut + vectorLoopCount), pack8);
    }
#endif
    for (; vectorLoopCount < HISTOGRAM_BINS; vectorLoopCount++) {
        Rpp32f eqVal =
            (static_cast<Rpp32f>(cdf[vectorLoopCount]) - static_cast<Rpp32f>(minCdf)) * multScalar;
        eqVal = std::min(std::max(eqVal, 0.0f), 255.0f);
        lut[vectorLoopCount] = static_cast<Rpp8u>(std::round(eqVal));
    }
}

// Apply LUT to image buffer
inline void apply_lut_tensor(const Rpp8u* src, Rpp8u* dst, Rpp32u roiWidth, Rpp32u roiHeight,
                             const Rpp8u* lut, Rpp32u srcRowStride, Rpp32u dstRowStride) {
    for (Rpp32u y = 0; y < roiHeight; y++) {
        const Rpp8u* srcRow = src + y * srcRowStride;
        Rpp8u* dstRow = dst + y * dstRowStride;
        for (Rpp32u x = 0; x < roiWidth; x++) dstRow[x] = lut[srcRow[x]];
    }
}

// Histogram equalization on Y channel
inline void histogram_equalize_host_compute(const Rpp8u* srcY, Rpp8u* dstY, Rpp32u roiWidth,
                                            Rpp32u roiHeight, Rpp32u pixels) {
    Rpp32u hist[HISTOGRAM_BINS] = {0};
    Rpp8u lut[HISTOGRAM_BINS];

    collect_hist_y_buffer(srcY, hist, pixels);
    build_lut_from_hist_host(hist, lut, pixels);
    apply_lut_tensor(srcY, dstY, roiWidth, roiHeight, lut, roiWidth, roiWidth);
}

RppStatus histogram_equalize_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                               RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType, RppLayoutParams layoutParams,
                                               rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full(static_cast<Rpp32s>(srcDescPtr->w),
                                                static_cast<Rpp32s>(srcDescPtr->h));
    Rpp32u numThreads = handle.GetNumThreads();

    // Pre-allocate YCbCr buffers for 3-channel images outside the parallel loop
    // Using max possible size (srcDescPtr->w * srcDescPtr->h) to handle any ROI
    Rpp8u* yBufBatch = nullptr;
    Rpp8u* cbBufBatch = nullptr;
    Rpp8u* crBufBatch = nullptr;
    Rpp32u maxPixelsPerImage = srcDescPtr->w * srcDescPtr->h;

    if (srcDescPtr->c == 3) {
        // Allocate buffers for each thread to avoid race conditions
        yBufBatch = static_cast<Rpp8u*>(malloc(numThreads * maxPixelsPerImage * sizeof(Rpp8u)));
        cbBufBatch = static_cast<Rpp8u*>(malloc(numThreads * maxPixelsPerImage * sizeof(Rpp8u)));
        crBufBatch = static_cast<Rpp8u*>(malloc(numThreads * maxPixelsPerImage * sizeof(Rpp8u)));

        // Check for allocation failures
        if (!yBufBatch || !cbBufBatch || !crBufBatch) {
            free(yBufBatch);
            free(cbBufBatch);
            free(crBufBatch);
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
        }
    }

    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(numThreads));
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u roiWidth = roi.xywhROI.roiWidth;
        Rpp32u roiHeight = roi.xywhROI.roiHeight;
        Rpp32u pixels = roiWidth * roiHeight;

        // Get thread-local buffer pointers for 3-channel images
        Rpp8u* yBuf = nullptr;
        Rpp8u* cbBuf = nullptr;
        Rpp8u* crBuf = nullptr;
        Rpp8u* dstYBuf = nullptr;

        if (srcDescPtr->c == 3) {
            int threadId = omp_get_thread_num();
            Rpp32u threadOffset = threadId * maxPixelsPerImage;
            yBuf = yBufBatch + threadOffset;
            cbBuf = cbBufBatch + threadOffset;
            crBuf = crBufBatch + threadOffset;
            dstYBuf = yBuf;
        }

#if __AVX2__
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32u alignedLength =
            (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
#endif

        // Histogram equalize without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB,
                *yPtr, *cbPtr, *crPtr;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(yPtr, cbPtr, crPtr, p);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    rgb_to_ycbcr_compute(srcPtrTempR++, srcPtrTempG++, srcPtrTempB++, yPtr++,
                                         cbPtr++, crPtr++);

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTempR++, dstPtrTempG++,
                                         dstPtrTempB++);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Histogram equalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow, *yPtr, *cbPtr, *crPtr;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(yPtr, cbPtr, crPtr, p);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    rgb_to_ycbcr_compute(srcPtrTempR++, srcPtrTempG++, srcPtrTempB++, yPtr++,
                                         cbPtr++, crPtr++);

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_store48_f32pln3_to_u8pkd3_avx(dstPtrTemp, p);

                    dstPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++) {
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTemp, dstPtrTemp + 1,
                                         dstPtrTemp + 2);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Histogram equalize with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB, *yPtr, *cbPtr, *crPtr;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;
#if __AVX2__
            if (roi.xywhROI.roiWidth >= vectorIncrement)
                alignedLength = ((roi.xywhROI.roiWidth / vectorIncrement) - 1) * vectorIncrement;
            else
                alignedLength = 0;
#endif
            for (int i = 0; i < roiHeight; i++) {
                Rpp8u* srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, yPtr, cbPtr, crPtr, p);

                    srcPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++) {
                    rgb_to_ycbcr_compute(srcPtrTemp, srcPtrTemp + 1, srcPtrTemp + 2, yPtr++,
                                         cbPtr++, crPtr++);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);

                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTempR++, dstPtrTempG++,
                                         dstPtrTempB++);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Histogram equalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRow, *dstPtrRow, *yPtr, *cbPtr, *crPtr;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;
#if __AVX2__
            if (roi.xywhROI.roiWidth >= vectorIncrement)
                alignedLength = ((roi.xywhROI.roiWidth / vectorIncrement) - 1) * vectorIncrement;
            else
                alignedLength = 0;
#endif
            for (int i = 0; i < roiHeight; i++) {
                Rpp8u* srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, yPtr, cbPtr, crPtr, p);

                    srcPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++) {
                    rgb_to_ycbcr_compute(srcPtrTemp, srcPtrTemp + 1, srcPtrTemp + 2, yPtr++,
                                         cbPtr++, crPtr++);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for (int i = 0; i < roiHeight; i++) {
                Rpp8u* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);

                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++) {
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTemp, dstPtrTemp + 1,
                                         dstPtrTemp + 2);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Histogram equalize for single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32u hist[HISTOGRAM_BINS];
            memset(hist, 0, HISTOGRAM_BINS * sizeof(Rpp32u));
            Rpp8u lutBatch[HISTOGRAM_BINS];
            Rpp8u *srcPtr, *dstPtr;
            srcPtr = srcPtrChannel;
            dstPtr = dstPtrChannel;
            collect_hist_pln_tensor_host(srcPtr, hist, roiWidth, roiHeight,
                                         srcDescPtr->strides.hStride);
            build_lut_from_hist_host(hist, lutBatch, pixels);
            apply_lut_tensor(srcPtr, dstPtr, roiWidth, roiHeight, lutBatch,
                             srcDescPtr->strides.hStride, dstDescPtr->strides.hStride);
        }
    }

    // Free pre-allocated batch buffers after parallel processing
    free(yBufBatch);
    free(cbBufBatch);
    free(crBufBatch);

    return RPP_SUCCESS;
}

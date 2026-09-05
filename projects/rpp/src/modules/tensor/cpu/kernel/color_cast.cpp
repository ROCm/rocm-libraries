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

inline void compute_color_cast_48_host(__m128* p, __m128 pMul, __m128* pAdd) {
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[3] = _mm_fmadd_ps(_mm_sub_ps(p[3], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[4] = _mm_fmadd_ps(_mm_sub_ps(p[4], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[5] = _mm_fmadd_ps(_mm_sub_ps(p[5], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[6] = _mm_fmadd_ps(_mm_sub_ps(p[6], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[7] = _mm_fmadd_ps(_mm_sub_ps(p[7], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[8] = _mm_fmadd_ps(_mm_sub_ps(p[8], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[9] = _mm_fmadd_ps(_mm_sub_ps(p[9], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[10] = _mm_fmadd_ps(_mm_sub_ps(p[10], pAdd[2]), pMul, pAdd[2]);  // color_cast adjustment Bs
    p[11] = _mm_fmadd_ps(_mm_sub_ps(p[11], pAdd[2]), pMul, pAdd[2]);  // color_cast adjustment Bs
}

inline void compute_color_cast_12_host(__m128* p, __m128 pMul, __m128* pAdd) {
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);  // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[1]), pMul, pAdd[1]);  // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[2]), pMul, pAdd[2]);  // color_cast adjustment Rs
}

inline void compute_color_cast_24_host(__m256* p, __m256 pMul, __m256* pAdd) {
    p[0] =
        _mm256_fmadd_ps(_mm256_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);  // color_cast adjustment Rs
    p[1] =
        _mm256_fmadd_ps(_mm256_sub_ps(p[1], pAdd[1]), pMul, pAdd[1]);  // color_cast adjustment Gs
    p[2] =
        _mm256_fmadd_ps(_mm256_sub_ps(p[2], pAdd[2]), pMul, pAdd[2]);  // color_cast adjustment Bs
}

// Helper function for u8->u8 color_cast processing
inline void color_cast_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp8u* dstPtrImage, RpptDescPtr dstDescPtr, Rpp32f rParam,
                                       Rpp32f gParam, Rpp32f bParam, Rpp32f alphaParam, RpptROI roi,
                                       RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    __m128 pMul = _mm_set1_ps(alphaParam);
    __m128 pAdd[3];
    pAdd[0] = _mm_set1_ps(bParam);
    pAdd[1] = _mm_set1_ps(gParam);
    pAdd[2] = _mm_set1_ps(rParam);

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Color Cast with fused output-layout toggle (NHWC -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m128 p[12];

                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[0] - bParam)) + bParam));
                *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[1] - gParam)) + gParam));
                *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[2] - rParam)) + rParam));

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Color Cast with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8u srcPtrTempR_local[16], srcPtrTempG_local[16], srcPtrTempB_local[16],
                    dstPtrTemp_local[48];
                for (int cnt = 0; cnt < 16; cnt++) {
                    srcPtrTempR_local[cnt] = srcPtrTempR[cnt];
                    srcPtrTempG_local[cnt] = srcPtrTempG[cnt];
                    srcPtrTempB_local[cnt] = srcPtrTempB[cnt];
                }

                __m128 p[12];

                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR_local, srcPtrTempG_local,
                              srcPtrTempB_local, p);        // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 48; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempR - bParam)) + bParam));
                dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempG - gParam)) + gParam));
                dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempB - rParam)) + rParam));

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8u *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                Rpp8u srcPtrTemp_local[48], dstPtrTemp_local[48];
                for (int cnt = 0; cnt < 48; cnt++) srcPtrTemp_local[cnt] = srcPtrTemp[cnt];

                __m128 p[12];

                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp_local, p);  // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 48; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[0] - bParam)) + bParam));
                dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[1] - gParam)) + gParam));
                dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (srcPtrTemp[2] - rParam)) + rParam));

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NCHW -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128 p[12];

                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                           // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempR - bParam)) + bParam));
                *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempG - gParam)) + gParam));
                *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf((alphaParam * (*srcPtrTempB - rParam)) + rParam));

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }
}

RppStatus color_cast_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptRGB* rgbTensor,
                                       Rpp32f* alphaTensor, RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType, RppLayoutParams layoutParams,
                                       rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f rParam = rgbTensor[batchCount].R;
        Rpp32f gParam = rgbTensor[batchCount].G;
        Rpp32f bParam = rgbTensor[batchCount].B;
        Rpp32f alphaParam = alphaTensor[batchCount];

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        color_cast_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, rParam, gParam,
                                   bParam, alphaParam, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for f32->f32 color_cast processing
inline void color_cast_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                         Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr, Rpp32f rParam,
                                         Rpp32f gParam, Rpp32f bParam, Rpp32f alphaParam,
                                         RpptROI roi, RppLayoutParams layoutParams,
                                         Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u alignedLength = (bufferLength / 12) * 12;
    Rpp32u vectorIncrement = 12;
    Rpp32u vectorIncrementPerChannel = 4;

    __m128 pMul = _mm_set1_ps(alphaParam);
    __m128 pAdd[3];
    pAdd[0] = _mm_set1_ps(bParam);
    pAdd[1] = _mm_set1_ps(gParam);
    pAdd[2] = _mm_set1_ps(rParam);

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Color Cast with fused output-layout toggle (NHWC -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f32
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[0] - bParam)) + bParam);
                *dstPtrTempG = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[1] - gParam)) + gParam);
                *dstPtrTempB = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[2] - rParam)) + rParam);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Color Cast with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                // Use intermediate buffers like F16 does - note: dst needs 13 elements for SIMD
                // overrun protection
                Rpp32f srcPtrTempR_local[4], srcPtrTempG_local[4], srcPtrTempB_local[4],
                    dstPtrTemp_local[13];

                for (int cnt = 0; cnt < 4; cnt++) {
                    srcPtrTempR_local[cnt] = srcPtrTempR[cnt];
                    srcPtrTempG_local[cnt] = srcPtrTempG[cnt];
                    srcPtrTempB_local[cnt] = srcPtrTempB[cnt];
                }

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_local, srcPtrTempG_local,
                              srcPtrTempB_local, p);        // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f32
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 12; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempR - bParam)) + bParam);
                dstPtrTemp[1] = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempG - gParam)) + gParam);
                dstPtrTemp[2] = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempB - rParam)) + rParam);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                Rpp32f srcPtrTemp_local[12], dstPtrTemp_local[12];
                for (int cnt = 0; cnt < 12; cnt++) srcPtrTemp_local[cnt] = srcPtrTemp[cnt];

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_local, p);  // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f32
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 12; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                dstPtrTemp[0] = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[0] - bParam)) + bParam);
                dstPtrTemp[1] = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[1] - gParam)) + gParam);
                dstPtrTemp[2] = RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[2] - rParam)) + rParam);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NCHW -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                           // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f32
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTempR = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempR - bParam)) + bParam);
                *dstPtrTempG = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempG - gParam)) + gParam);
                *dstPtrTempB = RPPPIXELCHECKF32((alphaParam * (*srcPtrTempB - rParam)) + rParam);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }
}

RppStatus color_cast_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                         RpptDescPtr dstDescPtr, RpptRGB* rgbTensor,
                                         Rpp32f* alphaTensor, RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType, RppLayoutParams layoutParams,
                                         rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f rParam = rgbTensor[batchCount].R * ONE_OVER_255;
        Rpp32f gParam = rgbTensor[batchCount].G * ONE_OVER_255;
        Rpp32f bParam = rgbTensor[batchCount].B * ONE_OVER_255;
        Rpp32f alphaParam = alphaTensor[batchCount];

        Rpp32f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        color_cast_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, rParam,
                                     gParam, bParam, alphaParam, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for f16->f16 color_cast processing
inline void color_cast_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                         Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr, Rpp32f rParam,
                                         Rpp32f gParam, Rpp32f bParam, Rpp32f alphaParam,
                                         RpptROI roi, RppLayoutParams layoutParams,
                                         Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

#if __AVX2__
    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pMul;
    pMul = _mm256_set1_ps(alphaParam);
    __m256 pAdd[3];
    pAdd[0] = _mm256_set1_ps(bParam);
    pAdd[1] = _mm256_set1_ps(gParam);
    pAdd[2] = _mm256_set1_ps(rParam);
#else
    Rpp32u alignedLength = (bufferLength / 12) * 12;
    Rpp32u vectorIncrement = 12;
    Rpp32u vectorIncrementPerChannel = 4;
    __m128 pMul = _mm_set1_ps(alphaParam);
    __m128 pAdd[3];
    pAdd[0] = _mm_set1_ps(bParam);
    pAdd[1] = _mm_set1_ps(gParam);
    pAdd[2] = _mm_set1_ps(rParam);
#endif

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Color Cast with fused output-layout toggle (NHWC -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[4];

                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_color_cast_24_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                for (int cnt = 0; cnt < 12; cnt++)
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) * (srcPtrTemp + cnt);

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);  // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4,
                               dstPtrTemp_ps + 8, p);  // simd stores

                for (int cnt = 0; cnt < 4; cnt++) {
                    *(dstPtrTempR + cnt) = (Rpp16f) * (dstPtrTemp_ps + cnt);
                    *(dstPtrTempG + cnt) = (Rpp16f) * (dstPtrTemp_ps + 4 + cnt);
                    *(dstPtrTempB + cnt) = (Rpp16f) * (dstPtrTemp_ps + 8 + cnt);
                }
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[0] - bParam)) + bParam);
                *dstPtrTempG =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[1] - gParam)) + gParam);
                *dstPtrTempB =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[2] - rParam)) + rParam);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Color Cast with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[4];

                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);              // simd loads
                compute_color_cast_24_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                               p);  // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                for (int cnt = 0; cnt < 4; cnt++) {
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) * (srcPtrTempR + cnt);
                    *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) * (srcPtrTempG + cnt);
                    *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) * (srcPtrTempB + cnt);
                }

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4,
                              srcPtrTemp_ps + 8, p);        // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps,
                               p);  // simd stores

                for (int cnt = 0; cnt < 12; cnt++)
                    *(dstPtrTemp + cnt) = (Rpp16f) * (dstPtrTemp_ps + cnt);
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempR - bParam)) + bParam);
                dstPtrTemp[1] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempG - gParam)) + gParam);
                dstPtrTemp[2] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempB - rParam)) + rParam);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp16f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
#if __AVX2__
                __m256 p[4];

                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_color_cast_24_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                               p);  // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                for (int cnt = 0; cnt < 12; cnt++)
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) * (srcPtrTemp + cnt);

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);  // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps,
                               p);  // simd stores

                for (int cnt = 0; cnt < 12; cnt++)
                    *(dstPtrTemp + cnt) = (Rpp16f) * (dstPtrTemp_ps + cnt);
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                dstPtrTemp[0] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[0] - bParam)) + bParam);
                dstPtrTemp[1] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[1] - gParam)) + gParam);
                dstPtrTemp[2] =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (srcPtrTemp[2] - rParam)) + rParam);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NCHW -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                __m256 p[4];

                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);              // simd loads
                compute_color_cast_24_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                for (int cnt = 0; cnt < 4; cnt++) {
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) * (srcPtrTempR + cnt);
                    *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) * (srcPtrTempG + cnt);
                    *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) * (srcPtrTempB + cnt);
                }

                __m128 p[4];

                rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4,
                              srcPtrTemp_ps + 8, p);        // simd loads
                compute_color_cast_12_host(p, pMul, pAdd);  // color_cast adjustment
                // boundary checks for f16
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4,
                               dstPtrTemp_ps + 8, p);  // simd stores

                for (int cnt = 0; cnt < 4; cnt++) {
                    *(dstPtrTempR + cnt) = (Rpp16f) * (dstPtrTemp_ps + cnt);
                    *(dstPtrTempG + cnt) = (Rpp16f) * (dstPtrTemp_ps + 4 + cnt);
                    *(dstPtrTempB + cnt) = (Rpp16f) * (dstPtrTemp_ps + 8 + cnt);
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
                *dstPtrTempR =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempR - bParam)) + bParam);
                *dstPtrTempG =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempG - gParam)) + gParam);
                *dstPtrTempB =
                    (Rpp16f)RPPPIXELCHECKF32((alphaParam * (*srcPtrTempB - rParam)) + rParam);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }
}

RppStatus color_cast_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                         RpptDescPtr dstDescPtr, RpptRGB* rgbTensor,
                                         Rpp32f* alphaTensor, RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType, RppLayoutParams layoutParams,
                                         rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f rParam = rgbTensor[batchCount].R * ONE_OVER_255;
        Rpp32f gParam = rgbTensor[batchCount].G * ONE_OVER_255;
        Rpp32f bParam = rgbTensor[batchCount].B * ONE_OVER_255;
        Rpp32f alphaParam = alphaTensor[batchCount];

        Rpp16f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        color_cast_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, rParam,
                                     gParam, bParam, alphaParam, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for i8->i8 color_cast processing
inline void color_cast_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp8s* dstPtrImage, RpptDescPtr dstDescPtr, Rpp32f rParam,
                                       Rpp32f gParam, Rpp32f bParam, Rpp32f alphaParam, RpptROI roi,
                                       RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    __m128 pMul = _mm_set1_ps(alphaParam);
    __m128 pAdd[3];
    pAdd[0] = _mm_set1_ps(bParam);
    pAdd[1] = _mm_set1_ps(gParam);
    pAdd[2] = _mm_set1_ps(rParam);

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Color Cast with fused output-layout toggle (NHWC -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m128 p[12];

                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);  // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f srcPtrTempI8[3];
                srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                *dstPtrTempR = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[0] - bParam)) +
                                                      bParam - 128);
                *dstPtrTempG = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[1] - gParam)) +
                                                      gParam - 128);
                *dstPtrTempB = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[2] - rParam)) +
                                                      rParam - 128);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Color Cast with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR + i * srcDescPtr->strides.hStride;
            srcPtrTempG = srcPtrRowG + i * srcDescPtr->strides.hStride;
            srcPtrTempB = srcPtrRowB + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8s srcPtrTempR_local[16], srcPtrTempG_local[16], srcPtrTempB_local[16],
                    dstPtrTemp_local[48];
                for (int cnt = 0; cnt < 16; cnt++) {
                    srcPtrTempR_local[cnt] = srcPtrTempR[cnt];
                    srcPtrTempG_local[cnt] = srcPtrTempG[cnt];
                    srcPtrTempB_local[cnt] = srcPtrTempB[cnt];
                }

                __m128 p[12];

                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR_local, srcPtrTempG_local,
                              srcPtrTempB_local, p);        // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 48; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f srcPtrTempI8[3];
                srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[0] - bParam)) +
                                                       bParam - 128);
                dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[1] - gParam)) +
                                                       gParam - 128);
                dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[2] - rParam)) +
                                                       rParam - 128);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp8s *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
            dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                Rpp8s srcPtrTemp_local[48], dstPtrTemp_local[48];
                for (int cnt = 0; cnt < 48; cnt++) srcPtrTemp_local[cnt] = srcPtrTemp[cnt];

                __m128 p[12];

                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp_local, p);  // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp_local, p);  // simd stores

                for (int cnt = 0; cnt < 48; cnt++) dstPtrTemp[cnt] = dstPtrTemp_local[cnt];

                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                Rpp32f srcPtrTempI8[3];
                srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[0] - bParam)) +
                                                       bParam - 128);
                dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[1] - gParam)) +
                                                       gParam - 128);
                dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[2] - rParam)) +
                                                       rParam - 128);

                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
        }
    }

    // Color Cast without fused output-layout toggle (NCHW -> NCHW)
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

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128 p[12];

                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                           // simd loads
                compute_color_cast_48_host(p, pMul, pAdd);  // color_cast adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                Rpp32f srcPtrTempI8[3];
                srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                *dstPtrTempR = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[0] - bParam)) +
                                                      bParam - 128);
                *dstPtrTempG = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[1] - gParam)) +
                                                      gParam - 128);
                *dstPtrTempB = (Rpp8s)RPPPIXELCHECKI8((alphaParam * (srcPtrTempI8[2] - rParam)) +
                                                      rParam - 128);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }
}

RppStatus color_cast_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptRGB* rgbTensor,
                                       Rpp32f* alphaTensor, RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType, RppLayoutParams layoutParams,
                                       rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f rParam = rgbTensor[batchCount].R;
        Rpp32f gParam = rgbTensor[batchCount].G;
        Rpp32f bParam = rgbTensor[batchCount].B;
        Rpp32f alphaParam = alphaTensor[batchCount];

        Rpp8s* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        color_cast_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, rParam, gParam,
                                   bParam, alphaParam, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

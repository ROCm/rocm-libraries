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

inline void compute_exposure_48_host(__m256* p, __m256& pExposureParam) {
    p[0] = _mm256_mul_ps(p[0], pExposureParam);  // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);  // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);  // exposure adjustment
    p[3] = _mm256_mul_ps(p[3], pExposureParam);  // exposure adjustment
    p[4] = _mm256_mul_ps(p[4], pExposureParam);  // exposure adjustment
    p[5] = _mm256_mul_ps(p[5], pExposureParam);  // exposure adjustment
}

inline void compute_exposure_24_host(__m256* p, __m256& pExposureParam) {
    p[0] = _mm256_mul_ps(p[0], pExposureParam);  // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);  // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);  // exposure adjustment
}

inline void compute_exposure_16_host(__m256* p, __m256& pExposureParam) {
    p[0] = _mm256_mul_ps(p[0], pExposureParam);  // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);  // exposure adjustment
}

inline void compute_exposure_8_host(__m256* p, __m256& pExposureParam) {
    p[0] = _mm256_mul_ps(p[0], pExposureParam);  // exposure adjustment
}

// Helper function for u8->u8 exposure processing
inline void exposure_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8u* dstPtrImage,
                                     RpptDescPtr dstDescPtr, Rpp32f multiplyingFactor, RpptROI roi,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    __m256 pExposureParam;
    pExposureParam = _mm256_set1_ps(multiplyingFactor);

    // Exposure with fused output-layout toggle (NHWC -> NCHW)
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
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_exposure_48_host(p, pExposureParam);  // exposure adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(srcPtrTemp[0])) * multiplyingFactor));
                *dstPtrTempG = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(srcPtrTemp[1])) * multiplyingFactor));
                *dstPtrTempB = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(srcPtrTemp[2])) * multiplyingFactor));

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Exposure with fused output-layout toggle (NCHW -> NHWC)
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
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);                // simd loads
                compute_exposure_48_host(p, pExposureParam);  // exposure adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(*srcPtrTempR)) * multiplyingFactor));
                dstPtrTemp[1] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(*srcPtrTempG)) * multiplyingFactor));
                dstPtrTemp[2] = (Rpp8u)RPPPIXELCHECK(
                    std::nearbyintf(((Rpp32f)(*srcPtrTempB)) * multiplyingFactor));

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Exposure without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = bufferLength & ~15;
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
                dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);  // simd loads
                    compute_exposure_16_host(p, pExposureParam);             // exposure adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = (Rpp8u)RPPPIXELCHECK(
                        std::nearbyintf(((Rpp32f)(*srcPtrTemp)) * multiplyingFactor));
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }
}

RppStatus exposure_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* exposureFactorTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f exposureFactor = exposureFactorTensor[batchCount];
        Rpp32f multiplyingFactor = pow(2, exposureFactor);

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        exposure_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 multiplyingFactor, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for f32->f32 exposure processing
inline void exposure_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                       Rpp32f multiplyingFactor, RpptROI roi,
                                       RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pExposureParam;
    pExposureParam = _mm256_set1_ps(multiplyingFactor);

    // Exposure with fused output-layout toggle (NHWC -> NCHW)
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
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_exposure_24_host(p, pExposureParam);  // exposure adjustment
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR = RPPPIXELCHECKF32(srcPtrTemp[0] * multiplyingFactor);
                *dstPtrTempG = RPPPIXELCHECKF32(srcPtrTemp[1] * multiplyingFactor);
                *dstPtrTempB = RPPPIXELCHECKF32(srcPtrTemp[2] * multiplyingFactor);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Exposure with fused output-layout toggle (NCHW -> NHWC)
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
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);                // simd loads
                compute_exposure_24_host(p, pExposureParam);  // exposure adjustment
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR) * multiplyingFactor);
                dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG) * multiplyingFactor);
                dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB) * multiplyingFactor);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Exposure without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
                dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[1];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);  // simd loads
                    compute_exposure_8_host(p, pExposureParam);              // exposure adjustment
                    rpp_pixel_check_0to1(p, 1);
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp) * multiplyingFactor);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }
}

RppStatus exposure_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* exposureFactorTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f exposureFactor = exposureFactorTensor[batchCount];
        Rpp32f multiplyingFactor = pow(2, exposureFactor);

        Rpp32f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        exposure_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                   multiplyingFactor, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for f16->f16 exposure processing
inline void exposure_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                       Rpp32f multiplyingFactor, RpptROI roi,
                                       RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pExposureParam;
    pExposureParam = _mm256_set1_ps(multiplyingFactor);

    // Exposure with fused output-layout toggle (NHWC -> NCHW)
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
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_exposure_24_host(p, pExposureParam);  // exposure adjustment
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0] * multiplyingFactor);
                *dstPtrTempG = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1] * multiplyingFactor);
                *dstPtrTempB = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2] * multiplyingFactor);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Exposure with fused output-layout toggle (NCHW -> NHWC)
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
                __m256 p[3];
                rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);                // simd loads
                compute_exposure_24_host(p, pExposureParam);  // exposure adjustment
                rpp_pixel_check_0to1(p, 3);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] =
                    (Rpp16f)RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempR) * multiplyingFactor);
                dstPtrTemp[1] =
                    (Rpp16f)RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempG) * multiplyingFactor);
                dstPtrTemp[2] =
                    (Rpp16f)RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempB) * multiplyingFactor);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Exposure without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
                dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[1];
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, p);  // simd loads
                    compute_exposure_8_host(p, pExposureParam);              // exposure adjustment
                    rpp_pixel_check_0to1(p, 1);
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp =
                        (Rpp16f)RPPPIXELCHECKF32((Rpp32f)(*srcPtrTemp) * multiplyingFactor);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }
}

RppStatus exposure_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* exposureFactorTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f exposureFactor = exposureFactorTensor[batchCount];
        Rpp32f multiplyingFactor = pow(2, exposureFactor);

        Rpp16f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        exposure_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                   multiplyingFactor, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// Helper function for i8->i8 exposure processing
inline void exposure_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8s* dstPtrImage,
                                     RpptDescPtr dstDescPtr, Rpp32f multiplyingFactor, RpptROI roi,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    __m256 pExposureParam;
    pExposureParam = _mm256_set1_ps(multiplyingFactor);

    // Exposure with fused output-layout toggle (NHWC -> NCHW)
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
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                compute_exposure_48_host(p, pExposureParam);  // exposure adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR =
                    (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[0] + 128) * multiplyingFactor - 128);
                *dstPtrTempG =
                    (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[1] + 128) * multiplyingFactor - 128);
                *dstPtrTempB =
                    (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[2] + 128) * multiplyingFactor - 128);

                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Exposure with fused output-layout toggle (NCHW -> NHWC)
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
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                              srcPtrTempB, p);                // simd loads
                compute_exposure_48_host(p, pExposureParam);  // exposure adjustment
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                               p);  // simd stores

                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                dstPtrTemp[0] = (Rpp8s)RPPPIXELCHECKI8(
                    (Rpp32f)((*srcPtrTempR) + 128) * multiplyingFactor - 128);
                dstPtrTemp[1] = (Rpp8s)RPPPIXELCHECKI8(
                    (Rpp32f)((*srcPtrTempG) + 128) * multiplyingFactor - 128);
                dstPtrTemp[2] = (Rpp8s)RPPPIXELCHECKI8(
                    (Rpp32f)((*srcPtrTempB) + 128) * multiplyingFactor - 128);

                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }
        }
    }

    // Exposure without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = bufferLength & ~15;
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow + i * srcDescPtr->strides.hStride;
                dstPtrTemp = dstPtrRow + i * dstDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);  // simd loads
                    compute_exposure_16_host(p, pExposureParam);             // exposure adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = (Rpp8s)RPPPIXELCHECK(
                        (Rpp32f)((*srcPtrTemp) + 128) * multiplyingFactor - 128);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }
}

RppStatus exposure_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* exposureFactorTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32u intraThreads = GetIntraImageNumThreads(handle, dstDescPtr->n, srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (intraThreads == 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f exposureFactor = exposureFactorTensor[batchCount];
        Rpp32f multiplyingFactor = pow(2, exposureFactor);

        Rpp8s* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        exposure_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 multiplyingFactor, roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

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

// Solarize compute OF N SIMD registers based on threshold and max value.
template <int N>
inline void compute_solarize_host(__m256* p, __m256 pThresholdParam, __m256 pMax) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        __m256 mask = _mm256_cmp_ps(p[i], pThresholdParam, _CMP_GE_OQ);
        p[i] = _mm256_blendv_ps(p[i], _mm256_sub_ps(pMax, p[i]), mask);
    }
}

RppStatus solarize_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
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

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        const Rpp8u maxVal = 255;

        Rpp32f thresholdParam =
            std::round(thresholdTensor[batchCount] *
                       maxVal);  // scale normalized [0, 1] threshold to [0, 255] range
#if __AVX2__
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
        __m256 pxThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Solarize with fused output-layout toggle (NHWC -> NCHW)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_solarize_host<6>(p, pxThresholdParam,
                                             avx_p255);  // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (maxVal - *(srcPtrTemp))
                                                                       : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 1))
                                         : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 2))
                                         : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_solarize_host<6>(p, pxThresholdParam,
                                             avx_p255);  // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp++ =
                        (*srcPtrTempR >= thresholdParam) ? (maxVal - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ =
                        (*srcPtrTempG >= thresholdParam) ? (maxVal - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ =
                        (*srcPtrTempB >= thresholdParam) ? (maxVal - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else {
#if __AVX2__
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
#endif
            for (int c = 0; c < layoutParams.channelParam; c++) {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);
                        compute_solarize_host<2>(p, pxThresholdParam, avx_p255);
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
#endif
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (maxVal - (*srcPtrTemp))
                                                                        : *srcPtrTemp;
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus solarize_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
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

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        const Rpp32f maxVal = 1.0f;

        Rpp32f thresholdParam = thresholdTensor[batchCount];

#if __AVX2__
        const Rpp32u vectorIncrement = 24;
        const Rpp32u vectorIncrementPerChannel = 8;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
        __m256 pThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_solarize_host<3>(p, pThresholdParam, avx_p1);  // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (maxVal - *(srcPtrTemp))
                                                                       : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 1))
                                         : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 2))
                                         : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                         // simd loads
                    compute_solarize_host<3>(p, pThresholdParam, avx_p1);  // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp++ =
                        (*srcPtrTempR >= thresholdParam) ? (maxVal - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ =
                        (*srcPtrTempG >= thresholdParam) ? (maxVal - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ =
                        (*srcPtrTempB >= thresholdParam) ? (maxVal - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else {
#if __AVX2__
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
#endif
            for (int c = 0; c < layoutParams.channelParam; c++) {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                        __m256 p;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);  // simd loads
                        compute_solarize_host<1>(&p, pThresholdParam,
                                                 avx_p1);  // threshold adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);  // simd stores

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
#endif
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (maxVal - (*srcPtrTemp))
                                                                        : *srcPtrTemp;
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus solarize_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
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

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        const Rpp32f maxVal = 1.0f;

        Rpp32f thresholdParam = thresholdTensor[batchCount];

#if __AVX2__
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
        __m256 pThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_solarize_host<3>(p, pThresholdParam, avx_p1);  // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (maxVal - *(srcPtrTemp))
                                                                       : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 1))
                                         : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 2))
                                         : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);                         // simd loads
                    compute_solarize_host<3>(p, pThresholdParam, avx_p1);  // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp++ =
                        (*srcPtrTempR >= thresholdParam) ? (maxVal - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ =
                        (*srcPtrTempG >= thresholdParam) ? (maxVal - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ =
                        (*srcPtrTempB >= thresholdParam) ? (maxVal - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else {
#if __AVX2__
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
#endif
            for (int c = 0; c < layoutParams.channelParam; c++) {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                        __m256 p;
                        rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, &p);  // simd loads
                        compute_solarize_host<1>(&p, pThresholdParam,
                                                 avx_p1);  // threshold adjustment
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p);  // simd stores

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
#endif
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (maxVal - (*srcPtrTemp))
                                                                        : *srcPtrTemp;
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus solarize_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
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

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        const Rpp8s maxVal = -1;
        const Rpp32s offset = 128;

        Rpp32f thresholdParam =
            std::round(thresholdTensor[batchCount] *
                       255);  // scale normalized [0, 1] threshold to [-128, 127] range
#if __AVX2__
        const Rpp32u vectorIncrement = 48;
        const Rpp32u vectorIncrementPerChannel = 16;
        Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
        __m256 pxThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    compute_solarize_host<6>(p, pxThresholdParam, avx_p255);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTempR++ = ((*(srcPtrTemp) + offset) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp))
                                         : *(srcPtrTemp);
                    *dstPtrTempG++ = ((*(srcPtrTemp + 1) + offset) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 1))
                                         : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = ((*(srcPtrTemp + 2) + offset) >= thresholdParam)
                                         ? (maxVal - *(srcPtrTemp + 2))
                                         : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
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
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG,
                                  srcPtrTempB, p);  // simd loads
                    compute_solarize_host<6>(p, pxThresholdParam,
                                             avx_p255);  // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp++ = ((*srcPtrTempR + offset) >= thresholdParam)
                                        ? (maxVal - (*srcPtrTempR))
                                        : *srcPtrTempR;
                    *dstPtrTemp++ = ((*srcPtrTempG + offset) >= thresholdParam)
                                        ? (maxVal - (*srcPtrTempG))
                                        : *srcPtrTempG;
                    *dstPtrTemp++ = ((*srcPtrTempB + offset) >= thresholdParam)
                                        ? (maxVal - (*srcPtrTempB))
                                        : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else {
#if __AVX2__
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel - 1);
#endif
            for (int c = 0; c < layoutParams.channelParam; c++) {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength;
                         vectorLoopCount += vectorIncrementPerChannel) {
#if __AVX2__
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);
                        compute_solarize_host<2>(p, pxThresholdParam, avx_p255);
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
#endif
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        *dstPtrTemp++ = ((*srcPtrTemp + offset) >= thresholdParam)
                                            ? (maxVal - (*srcPtrTemp))
                                            : *srcPtrTemp;
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

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

inline void compute_src_loc(int row, int col, Rpp32s* locArray, RpptDescPtr srcDescPtr,
                            RpptChannelOffsets* rgbOffset, RpptROI roi, int channelValue) {
    int xR, yR, xG, yG, xB, yB;
    xR = col + rgbOffset->r.x;
    yR = row + rgbOffset->r.y;
    xG = col + rgbOffset->g.x;
    yG = row + rgbOffset->g.y;
    xB = col + rgbOffset->b.x;
    yB = row + rgbOffset->b.y;

    int maxX = roi.xywhROI.xy.x + roi.xywhROI.roiWidth;
    int maxY = roi.xywhROI.xy.y + roi.xywhROI.roiHeight;

    if (xR >= maxX || xR < roi.xywhROI.xy.x || yR >= maxY || yR < roi.xywhROI.xy.y) {
        xR = col;
        yR = row;
    }

    if (xG >= maxX || xG < roi.xywhROI.xy.x || yG >= maxY || yG < roi.xywhROI.xy.y) {
        xG = col;
        yG = row;
    }

    if (xB >= maxX || xB < roi.xywhROI.xy.x || yB >= maxY || yB < roi.xywhROI.xy.y) {
        xB = col;
        yB = row;
    }

    locArray[0] = yR * srcDescPtr->strides.hStride + xR * channelValue;
    locArray[1] = yG * srcDescPtr->strides.hStride + xG * channelValue;
    locArray[2] = yB * srcDescPtr->strides.hStride + xB * channelValue;
}

inline void glitch_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8u* dstPtrImage,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffset,
                                   RpptROI roi, RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u vectorIncrement, alignedLength;

    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
        vectorIncrement = 8;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8u* dstRowPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstRowPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstRowPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load24_u8pkd3_to_f32pln3_avx, srcPtrChannel, p,
                              glitchSrcLocArray);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstRowPtrTempR, dstRowPtrTempG,
                               dstRowPtrTempB, p);

                dstRowPtrTempR += vectorIncrement;
                dstRowPtrTempG += vectorIncrement;
                dstRowPtrTempB += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        vectorIncrement = 16;
        Rpp32u vectorIncrementPkd = 48;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                Rpp32u rLoc = glitchSrcLocArray[0];
                Rpp32u gLoc = srcDescPtr->strides.cStride + glitchSrcLocArray[1];
                Rpp32u bLoc = 2 * srcDescPtr->strides.cStride + glitchSrcLocArray[2];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrChannel + rLoc,
                              srcPtrChannel + gLoc, srcPtrChannel + bLoc, p);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 3;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        vectorIncrement = 32;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++) {
                    __m256i p;
                    p = _mm256_loadu_si256(
                        (__m256i*)(srcPtrChannel +
                                   (glitchSrcLocArray[c] + (c * srcDescPtr->strides.cStride))));
                    _mm256_storeu_si256((__m256i*)(dstPtrTemp + (c * srcDescPtr->strides.cStride)),
                                        p);
                }
                dstPtrTemp += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c * dstDescPtr->strides.cStride) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 1;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        Rpp32u vectorIncrement = 10;
        Rpp32u vectorIncrementPkd = 30;
        Rpp32u numVectors = roi.xywhROI.roiWidth / vectorIncrement;
        Rpp32u alignedLength = (numVectors > 1) ? ((numVectors - 1) * vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256i p;
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load30_u8pkd3_to_u8pkd3_avx, srcPtrChannel,
                              glitchSrcLocArray, p);
                _mm256_storeu_si256((__m256i*)(dstPtrTemp), p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                for (int c = 0; c < 3; c++)
                    *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
            }
        }
    }
}

RppStatus glitch_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        glitch_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                               &rgbOffsets[batchCount], roi, layoutParams, intraThreads);
    }
    return RPP_SUCCESS;
}

inline void glitch_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                     Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets* rgbOffset, RpptROI roi,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u vectorIncrement, alignedLength;

    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
        vectorIncrement = 8;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp32f* dstRowPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstRowPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstRowPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load24_f32pkd3_to_f32pln3_avx, srcPtrChannel, p,
                              glitchSrcLocArray);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstRowPtrTempR, dstRowPtrTempG,
                               dstRowPtrTempB, p);

                dstRowPtrTempR += vectorIncrement;
                dstRowPtrTempG += vectorIncrement;
                dstRowPtrTempB += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        vectorIncrement = 8;
        Rpp32u vectorIncrementPkd = 24;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                p[0] = _mm256_loadu_ps(srcPtrChannel + glitchSrcLocArray[0]);
                p[1] = _mm256_loadu_ps(srcPtrChannel + srcDescPtr->strides.cStride +
                                       glitchSrcLocArray[1]);
                p[2] = _mm256_loadu_ps(srcPtrChannel + 2 * srcDescPtr->strides.cStride +
                                       glitchSrcLocArray[2]);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 3;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32u vectorIncrement = 8;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++) {
                    __m256 p;
                    p = _mm256_loadu_ps(srcPtrChannel +
                                        (glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride));
                    _mm256_storeu_ps((dstPtrTemp + c * srcDescPtr->strides.cStride), p);
                }
                dstPtrTemp += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c * dstDescPtr->strides.cStride) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 1;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32u vectorIncrement = 2;
        Rpp32u vectorIncrementPkd = 6;
        alignedLength = (((roi.xywhROI.roiWidth) / vectorIncrementPkd) * vectorIncrementPkd) -
                        vectorIncrementPkd;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p;
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load6_f32pkd3_to_f32pkd3_avx, srcPtrChannel,
                              glitchSrcLocArray, p);
                _mm256_storeu_ps(dstPtrTemp, p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                for (int c = 0; c < 3; c++)
                    *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
            }
        }
    }
}

RppStatus glitch_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        glitch_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 &rgbOffsets[batchCount], roi, layoutParams, intraThreads);
    }
    return RPP_SUCCESS;
}

inline void glitch_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                     Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets* rgbOffset, RpptROI roi,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp16f* dstRowPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstRowPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstRowPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;
            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 3;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            for (int i = 0; i < roi.xywhROI.roiWidth; i++) {
                compute_src_loc(dstLocRow, i, glitchSrcLocArray, srcDescPtr, rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c * dstDescPtr->strides.cStride) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 1;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            for (int i = 0; i < roi.xywhROI.roiWidth; i++) {
                compute_src_loc(dstLocRow, i, glitchSrcLocArray, srcDescPtr, rgbOffset, roi, 3);
                for (int c = 0; c < 3; c++)
                    *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
            }
        }
    }
}

RppStatus glitch_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        glitch_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 &rgbOffsets[batchCount], roi, layoutParams, intraThreads);
    }
    return RPP_SUCCESS;
}

inline void glitch_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8s* dstPtrImage,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffset,
                                   RpptROI roi, RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u vectorIncrement, alignedLength;

    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
        vectorIncrement = 8;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8s* dstRowPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstRowPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstRowPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load24_i8pkd3_to_f32pln3_avx, srcPtrChannel, p,
                              glitchSrcLocArray);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstRowPtrTempR, dstRowPtrTempG,
                               dstRowPtrTempB, p);

                dstRowPtrTempR += vectorIncrement;
                dstRowPtrTempG += vectorIncrement;
                dstRowPtrTempB += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        vectorIncrement = 16;
        Rpp32u vectorIncrementPkd = 48;
        Rpp32u alignedWidth = roi.xywhROI.roiWidth & ~(vectorIncrement - 1);
        alignedLength = (alignedWidth >= vectorIncrement) ? (alignedWidth - vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                Rpp32u rLoc = glitchSrcLocArray[0];
                Rpp32u gLoc = srcDescPtr->strides.cStride + glitchSrcLocArray[1];
                Rpp32u bLoc = 2 * srcDescPtr->strides.cStride + glitchSrcLocArray[2];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrChannel + rLoc,
                              srcPtrChannel + gLoc, srcPtrChannel + bLoc, p);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 3;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        vectorIncrement = 32;
        Rpp32u numVectors = roi.xywhROI.roiWidth / vectorIncrement;
        alignedLength = (numVectors > 1) ? ((numVectors - 1) * vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++) {
                    __m256i p;
                    p = _mm256_loadu_si256(
                        (__m256i*)(srcPtrChannel +
                                   (glitchSrcLocArray[c] + (c * srcDescPtr->strides.cStride))));
                    _mm256_storeu_si256((__m256i*)(dstPtrTemp + (c * srcDescPtr->strides.cStride)),
                                        p);
                }
                dstPtrTemp += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 1);
                for (int c = 0; c < 3; c++)
                    *(dstPtrTemp + c * dstDescPtr->strides.cStride) =
                        *(srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                dstPtrTemp += 1;
            }
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        vectorIncrement = 10;
        Rpp32u vectorIncrementPkd = 30;
        Rpp32u numVectors = roi.xywhROI.roiWidth / vectorIncrement;
        alignedLength = (numVectors > 1) ? ((numVectors - 1) * vectorIncrement) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s glitchSrcLocArray[3] = {0};
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256i p;
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                rpp_simd_load(rpp_glitch_load30_i8pkd3_to_i8pkd3_avx, srcPtrChannel,
                              glitchSrcLocArray, p);
                _mm256_storeu_si256((__m256i*)(dstPtrTemp), p);
                dstPtrTemp += vectorIncrementPkd;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                rgbOffset, roi, 3);
                for (int c = 0; c < 3; c++)
                    *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
            }
        }
    }
}

RppStatus glitch_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        glitch_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                               &rgbOffsets[batchCount], roi, layoutParams, intraThreads);
    }
    return RPP_SUCCESS;
}

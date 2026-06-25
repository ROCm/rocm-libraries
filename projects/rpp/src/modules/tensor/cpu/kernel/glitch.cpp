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
                            RpptChannelOffsets* rgbOffsets, RpptROI roi, int batchCount,
                            int channelValue) {
    int xR, yR, xG, yG, xB, yB;
    xR = col + rgbOffsets[batchCount].r.x;
    yR = row + rgbOffsets[batchCount].r.y;
    xG = col + rgbOffsets[batchCount].g.x;
    yG = row + rgbOffsets[batchCount].g.y;
    xB = col + rgbOffsets[batchCount].b.x;
    yB = row + rgbOffsets[batchCount].b.y;

    if (xR >= roi.xywhROI.roiWidth || xR < roi.xywhROI.xy.x || yR >= roi.xywhROI.roiHeight ||
        yR < roi.xywhROI.xy.y) {
        xR = col;
        yR = row;
    }

    if (xG >= roi.xywhROI.roiWidth || xG < roi.xywhROI.xy.x || yG >= roi.xywhROI.roiHeight ||
        yG < roi.xywhROI.xy.y) {
        xG = col;
        yG = row;
    }

    if (xB >= roi.xywhROI.roiWidth || xB < roi.xywhROI.xy.x || yB >= roi.xywhROI.roiHeight ||
        yB < roi.xywhROI.xy.y) {
        xB = col;
        yB = row;
    }

    locArray[0] = yR * srcDescPtr->strides.hStride + xR * channelValue;
    locArray[1] = yG * srcDescPtr->strides.hStride + xG * channelValue;
    locArray[2] = yB * srcDescPtr->strides.hStride + xB * channelValue;
}

RppStatus glitch_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
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

        Rpp32s glitchSrcLocArray[3] = {
            0};  // Since 3 destination pixels, one for each channel, are processed per iteration.

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement, alignedLength;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 8;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8u* dstRowPtrTempR = dstPtrRow;
                Rpp8u* dstRowPtrTempG = dstPtrRow + dstDescPtr->strides.cStride;
                Rpp8u* dstRowPtrTempB = dstPtrRow + 2 * dstDescPtr->strides.cStride;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load24_u8pkd3_to_f32pln3_avx, srcPtrChannel, p,
                                  glitchSrcLocArray);
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstRowPtrTempR,
                                   dstRowPtrTempG, dstRowPtrTempB, p);  // simd stores

                    dstRowPtrTempR += vectorIncrement;
                    dstRowPtrTempG += vectorIncrement;
                    dstRowPtrTempB += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                    *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                    *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 16;
            Rpp32u vectorIncrementPkd = 48;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8u* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    Rpp32u rLoc = glitchSrcLocArray[0];
                    Rpp32u gLoc = srcDescPtr->strides.cStride + glitchSrcLocArray[1];
                    Rpp32u bLoc = 2 * srcDescPtr->strides.cStride + glitchSrcLocArray[2];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrChannel + rLoc,
                                  srcPtrChannel + gLoc, srcPtrChannel + bLoc, p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c) = *(srcPtrChannel + glitchSrcLocArray[c] +
                                              c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 32;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8u* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++) {
                        __m256i p;
                        p = _mm256_loadu_si256(
                            (__m256i*)(srcPtrChannel +
                                       (glitchSrcLocArray[c] + (c * srcDescPtr->strides.cStride))));
                        _mm256_storeu_si256(
                            (__m256i*)(dstPtrTemp + (c * srcDescPtr->strides.cStride)), p);
                    }
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c * dstDescPtr->strides.cStride) = *(
                            srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 1;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32s vectorIncrement = 10;
            Rpp32s vectorIncrementPkd = 30;
            Rpp32u alignedLength = ((roi.xywhROI.roiWidth / vectorIncrement) - 1) * vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8u* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p;
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load30_u8pkd3_to_u8pkd3_avx, srcPtrChannel,
                                  glitchSrcLocArray, p);
                    _mm256_storeu_si256((__m256i*)(dstPtrTemp), p);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    for (int c = 0; c < 3; c++)
                        *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
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

        Rpp32s glitchSrcLocArray[3] = {
            0};  // Since 3 destination pixels, one for each channel, are processed per iteration.

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement, alignedLength;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 8;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp32f* dstRowPtrTempR = dstPtrRow;
                Rpp32f* dstRowPtrTempG = dstPtrRow + dstDescPtr->strides.cStride;
                Rpp32f* dstRowPtrTempB = dstPtrRow + 2 * dstDescPtr->strides.cStride;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load24_f32pkd3_to_f32pln3_avx, srcPtrChannel, p,
                                  glitchSrcLocArray);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstRowPtrTempR,
                                   dstRowPtrTempG, dstRowPtrTempB, p);  // simd stores

                    dstRowPtrTempR += vectorIncrement;
                    dstRowPtrTempG += vectorIncrement;
                    dstRowPtrTempB += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                    *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                    *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 8;
            Rpp32u vectorIncrementPkd = 24;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp32f* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    p[0] = _mm256_loadu_ps(srcPtrChannel + glitchSrcLocArray[0]);
                    p[1] = _mm256_loadu_ps(srcPtrChannel + srcDescPtr->strides.cStride +
                                           glitchSrcLocArray[1]);
                    p[2] = _mm256_loadu_ps(srcPtrChannel + 2 * srcDescPtr->strides.cStride +
                                           glitchSrcLocArray[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c) = *(srcPtrChannel + glitchSrcLocArray[c] +
                                              c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32u vectorIncrement = 8;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp32f* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++) {
                        __m256 p;
                        p = _mm256_loadu_ps(srcPtrChannel + (glitchSrcLocArray[c] +
                                                             c * srcDescPtr->strides.cStride));
                        _mm256_storeu_ps((dstPtrTemp + c * srcDescPtr->strides.cStride), p);
                    }
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c * dstDescPtr->strides.cStride) = *(
                            srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 1;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 2;  // 1 increment per channel (R, G, B) => 2 channels at a time
            Rpp32u vectorIncrementPkd =
                6;  // 3 channels per pixel (R, G, B) × 2 pixels at a time => 6
            alignedLength = (((roi.xywhROI.roiWidth) / vectorIncrementPkd) * vectorIncrementPkd) -
                            vectorIncrementPkd;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp32f* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p;
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load6_f32pkd3_to_f32pkd3_avx, srcPtrChannel,
                                  glitchSrcLocArray, p);
                    _mm256_storeu_ps(dstPtrTemp, p);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    for (int c = 0; c < 3; c++)
                        *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
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

        Rpp32s glitchSrcLocArray[3] = {
            0};  // Since 3 destination pixels, one for each channel, are processed per iteration.

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp16f* dstRowPtrTempR = dstPtrRow;
                Rpp16f* dstRowPtrTempG = dstPtrRow + dstDescPtr->strides.cStride;
                Rpp16f* dstRowPtrTempB = dstPtrRow + 2 * dstDescPtr->strides.cStride;
                for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                     vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                    *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                    *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp16f* dstPtrTemp = dstPtrRow;
                for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                     vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c) = *(srcPtrChannel + glitchSrcLocArray[c] +
                                              c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp16f* dstPtrTemp = dstPtrRow;
                for (int i = 0; i < roi.xywhROI.roiWidth; i++) {
                    compute_src_loc(dstLocRow, i, glitchSrcLocArray, srcDescPtr, rgbOffsets, roi,
                                    batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c * dstDescPtr->strides.cStride) = *(
                            srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 1;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp16f* dstPtrTemp = dstPtrRow;
                for (int i = 0; i < roi.xywhROI.roiWidth; i++) {
                    compute_src_loc(dstLocRow, i, glitchSrcLocArray, srcDescPtr, rgbOffsets, roi,
                                    batchCount, 3);
                    for (int c = 0; c < 3; c++)
                        *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
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

        Rpp32s glitchSrcLocArray[3] = {
            0};  // Since 3 destination pixels, one for each channel, are processed per iteration.

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement, alignedLength;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 8;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8s* dstRowPtrTempR = dstPtrRow;
                Rpp8s* dstRowPtrTempG = dstPtrRow + dstDescPtr->strides.cStride;
                Rpp8s* dstRowPtrTempB = dstPtrRow + 2 * dstDescPtr->strides.cStride;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load24_i8pkd3_to_f32pln3_avx, srcPtrChannel, p,
                                  glitchSrcLocArray);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstRowPtrTempR,
                                   dstRowPtrTempG, dstRowPtrTempB, p);  // simd stores

                    dstRowPtrTempR += vectorIncrement;
                    dstRowPtrTempG += vectorIncrement;
                    dstRowPtrTempB += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    *dstRowPtrTempR++ = *(srcPtrChannel + glitchSrcLocArray[0] + 0);
                    *dstRowPtrTempG++ = *(srcPtrChannel + glitchSrcLocArray[1] + 1);
                    *dstRowPtrTempB++ = *(srcPtrChannel + glitchSrcLocArray[2] + 2);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 16;
            Rpp32u vectorIncrementPkd = 48;
            alignedLength = (roi.xywhROI.roiWidth & ~(vectorIncrement - 1)) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8s* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    Rpp32u rLoc = glitchSrcLocArray[0];
                    Rpp32u gLoc = srcDescPtr->strides.cStride + glitchSrcLocArray[1];
                    Rpp32u bLoc = 2 * srcDescPtr->strides.cStride + glitchSrcLocArray[2];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrChannel + rLoc,
                                  srcPtrChannel + gLoc, srcPtrChannel + bLoc, p);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c) = *(srcPtrChannel + glitchSrcLocArray[c] +
                                              c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 32;
            alignedLength =
                (((roi.xywhROI.roiWidth) / vectorIncrement) * vectorIncrement) - vectorIncrement;

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8s* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++) {
                        __m256i p;
                        p = _mm256_loadu_si256(
                            (__m256i*)(srcPtrChannel +
                                       (glitchSrcLocArray[c] + (c * srcDescPtr->strides.cStride))));
                        _mm256_storeu_si256(
                            (__m256i*)(dstPtrTemp + (c * srcDescPtr->strides.cStride)), p);
                    }
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                        *(dstPtrTemp + c * dstDescPtr->strides.cStride) = *(
                            srcPtrChannel + glitchSrcLocArray[c] + c * srcDescPtr->strides.cStride);
                    dstPtrTemp += 1;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            vectorIncrement = 10;
            Rpp32s vectorIncrementPkd = 30;
            alignedLength =
                (((roi.xywhROI.roiWidth) / vectorIncrement) * vectorIncrement) - vectorIncrement;
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
                Rpp8s* dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p;
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load30_i8pkd3_to_i8pkd3_avx, srcPtrChannel,
                                  glitchSrcLocArray, p);
                    _mm256_storeu_si256((__m256i*)(dstPtrTemp), p);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                    compute_src_loc(dstLocRow, vectorLoopCount, glitchSrcLocArray, srcDescPtr,
                                    rgbOffsets, roi, batchCount, 3);
                    for (int c = 0; c < 3; c++)
                        *dstPtrTemp++ = *(srcPtrChannel + glitchSrcLocArray[c] + c);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

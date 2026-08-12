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

static inline RppStatus flip_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr,
                                             Rpp8u* dstPtrImage, RpptDescPtr dstDescPtr,
                                             Rpp32u horizontalFlag, Rpp32u verticalFlag,
                                             RpptROI roi, RppLayoutParams layoutParams) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp8u *srcPtrChannel, *dstPtrChannel;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    // Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src
    // pointer location)
    Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
    Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
    Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

    // Initialize load functions with default values
    auto load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_avx;
    auto load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_avx;
    auto load16Fn = &rpp_load16_u8_to_f32_avx;

    // Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags
    // enabled
    if (horizontalFlag == 1) {
        hFactor +=
            (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
        load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_mirror_avx;
        load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_mirror_avx;
        load16Fn = &rpp_load16_u8_to_f32_mirror_avx;
    }
    if (verticalFlag == 1) {
        vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
        hStrideSrcIncrement = -srcDescPtr->strides.hStride;
    }
    srcPtrChannel = srcPtrImage + vFactor + hFactor;

    // Compute constant increment, Decrement factors used in source pointer updation
    Rpp32s srcPtrIncrement = (horizontalFlag) ? -vectorIncrement : vectorIncrement;
    Rpp32u hFlipFactor = (vectorIncrement - 3) * horizontalFlag;
    Rpp32s srcPtrIncrementPerChannel =
        (horizontalFlag) ? -vectorIncrementPerChannel : vectorIncrementPerChannel;
    Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
    Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0
    // and verticalflag = 0)
    if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout)) {
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                memcpy(dstPtrRow, srcPtrRow, bufferLength);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    // flip with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];

                rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);  // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += srcPtrIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[0]));
                *dstPtrTempG++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[1]));
                *dstPtrTempB++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[2]));
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRowR += dstDescPtr->strides.hStride;
            dstPtrRowG += dstDescPtr->strides.hStride;
            dstPtrRowB += dstDescPtr->strides.hStride;
        }
    }

    // flip with fused output-layout toggle (NCHW -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 p[6];

                rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                                                  // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);  // simd stores

                srcPtrTempR += srcPtrIncrementPerChannel;
                srcPtrTempG += srcPtrIncrementPerChannel;
                srcPtrTempB += srcPtrIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }

            srcPtrTempR += hFlipFactorPerChannel;
            srcPtrTempG += hFlipFactorPerChannel;
            srcPtrTempB += hFlipFactorPerChannel;

            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(*srcPtrTempR));
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(*srcPtrTempG));
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(*srcPtrTempB));
                srcPtrTempR += srcPtrIncrementPerPixel;
                srcPtrTempG += srcPtrIncrementPerPixel;
                srcPtrTempB += srcPtrIncrementPerPixel;
            }

            srcPtrRowR += hStrideSrcIncrement;
            ;
            srcPtrRowG += hStrideSrcIncrement;
            srcPtrRowB += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];
                rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);                      // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);  // simd stores
                srcPtrTemp += srcPtrIncrement;
                dstPtrTemp += vectorIncrement;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[0]));
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[1]));
                *dstPtrTemp++ = (Rpp8u)RPPPIXELCHECK((Rpp32f)(srcPtrTemp[2]));
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = (bufferLength / 16) * 16;
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
                    __m256 p[2];
                    rpp_simd_load(load16Fn, srcPtrTemp, p);                    // simd loads
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactorPerChannel;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = (Rpp8u)RPPPIXELCHECK((Rpp32f)(*srcPtrTemp));
                    srcPtrTemp += srcPtrIncrementPerPixel;
                    dstPtrTemp++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                 Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType, RppLayoutParams layoutParams,
                                 rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        flip_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                             horizontalTensor[batchCount], verticalTensor[batchCount], roi,
                             layoutParams);
    }

    return RPP_SUCCESS;
}

static inline RppStatus flip_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                               Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                               Rpp32u horizontalFlag, Rpp32u verticalFlag,
                                               RpptROI roi, RppLayoutParams layoutParams) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32f *srcPtrChannel, *dstPtrChannel;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    // Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src
    // pointer location)
    Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
    Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
    Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;
    constexpr Rpp32u RGB_CHANNELS = 3;  // Number of channels in RGB processing.

    // The subtraction of elementsToSkip (typically RGB_CHANNELS - 1) from hFlipFactor
    // ensures correct alignment of the source pointer for the remaining elements.
    constexpr Rpp32u elementsToSkip =
        RGB_CHANNELS - 1;  // Skip two elements to align with the processing logic for RGB channels.

    // Initialize load functions with default values
    auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
    auto load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_avx;
    auto load8Fn = &rpp_load8_f32_to_f32_avx;

    // Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags
    // enabled
    if (horizontalFlag == 1) {
        hFactor +=
            (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
        load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
        load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
        load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
    }
    if (verticalFlag == 1) {
        vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
        hStrideSrcIncrement = -srcDescPtr->strides.hStride;
    }
    srcPtrChannel = srcPtrImage + vFactor + hFactor;

    // Compute constant increment, Decrement factors used in source pointer updation
    Rpp32s srcPtrIncrement = (horizontalFlag) ? -vectorIncrement : vectorIncrement;
    Rpp32u hFlipFactor = (vectorIncrement - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerChannel =
        (horizontalFlag) ? -vectorIncrementPerChannel : vectorIncrementPerChannel;
    Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
    Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0
    // and verticalflag = 0)
    if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout)) {
        Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp32f);
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    // flip with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];

                rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);  // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += srcPtrIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            srcPtrTemp += hFlipFactor - elementsToSkip;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR++ = srcPtrTemp[0];
                *dstPtrTempG++ = srcPtrTemp[1];
                *dstPtrTempB++ = srcPtrTemp[2];
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRowR += dstDescPtr->strides.hStride;
            dstPtrRowG += dstDescPtr->strides.hStride;
            dstPtrRowB += dstDescPtr->strides.hStride;
        }
    }

    // flip with fused output-layout toggle (NCHW -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 p[6];

                rpp_simd_load(load24FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                                                   // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);  // simd stores

                srcPtrTempR += srcPtrIncrementPerChannel;
                srcPtrTempG += srcPtrIncrementPerChannel;
                srcPtrTempB += srcPtrIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }

            srcPtrTempR += hFlipFactorPerChannel;
            srcPtrTempG += hFlipFactorPerChannel;
            srcPtrTempB += hFlipFactorPerChannel;

            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTemp++ = *srcPtrTempR;
                *dstPtrTemp++ = *srcPtrTempG;
                *dstPtrTemp++ = *srcPtrTempB;
                srcPtrTempR += srcPtrIncrementPerPixel;
                srcPtrTempG += srcPtrIncrementPerPixel;
                srcPtrTempB += srcPtrIncrementPerPixel;
            }

            srcPtrRowR += hStrideSrcIncrement;
            srcPtrRowG += hStrideSrcIncrement;
            srcPtrRowB += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];
                rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);                       // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);  // simd stores
                srcPtrTemp += srcPtrIncrement;
                dstPtrTemp += vectorIncrement;
            }
            srcPtrTemp += hFlipFactor - elementsToSkip;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTemp++ = srcPtrTemp[0];
                *dstPtrTemp++ = srcPtrTemp[1];
                *dstPtrTemp++ = srcPtrTemp[2];
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = (bufferLength / 8) * 8;
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
                    __m256 p[2];
                    rpp_simd_load(load8Fn, srcPtrTemp, p);                     // simd loads
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactorPerChannel;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp += srcPtrIncrementPerPixel;
                    dstPtrTemp++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                   Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType, RppLayoutParams layoutParams,
                                   rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        flip_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                               horizontalTensor[batchCount], verticalTensor[batchCount], roi,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

static inline RppStatus flip_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                               Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                               Rpp32u horizontalFlag, Rpp32u verticalFlag,
                                               RpptROI roi, RppLayoutParams layoutParams) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp16f *srcPtrChannel, *dstPtrChannel;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    // Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src
    // pointer location)
    Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
    Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
    Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

    // Initialize load functions with default values
    auto load24FnPkdPln = &rpp_load24_f16pkd3_to_f32pln3_avx;
    auto load24FnPlnPln = &rpp_load24_f16pln3_to_f32pln3_avx;
    auto load8Fn = &rpp_load8_f16_to_f32_avx;

    // Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags
    // enabled
    if (horizontalFlag == 1) {
        hFactor +=
            (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
        load24FnPkdPln = &rpp_load24_f16pkd3_to_f32pln3_mirror_avx;
        load24FnPlnPln = &rpp_load24_f16pln3_to_f32pln3_mirror_avx;
        load8Fn = &rpp_load8_f16_to_f32_mirror_avx;
    }
    if (verticalFlag == 1) {
        vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
        hStrideSrcIncrement = -srcDescPtr->strides.hStride;
    }
    srcPtrChannel = srcPtrImage + vFactor + hFactor;

    // Compute constant increment, Decrement factors used in source pointer updation
    Rpp32s srcPtrIncrement = (horizontalFlag) ? -vectorIncrement : vectorIncrement;
    Rpp32u hFlipFactor = (vectorIncrement - 3) * horizontalFlag;  // subtract 3 for RGB channels
    Rpp32s srcPtrIncrementPerChannel =
        (horizontalFlag) ? -vectorIncrementPerChannel : vectorIncrementPerChannel;
    Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
    Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0
    // and verticalflag = 0)
    if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout)) {
        Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp16f);
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    // flip with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];

                rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);  // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += srcPtrIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
                *dstPtrTempG++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
                *dstPtrTempB++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRowR += dstDescPtr->strides.hStride;
            dstPtrRowG += dstDescPtr->strides.hStride;
            dstPtrRowB += dstDescPtr->strides.hStride;
        }
    }

    // flip with fused output-layout toggle (NCHW -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 p[3];

                rpp_simd_load(load24FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                                                   // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);  // simd stores

                srcPtrTempR += srcPtrIncrementPerChannel;
                srcPtrTempG += srcPtrIncrementPerChannel;
                srcPtrTempB += srcPtrIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }

            srcPtrTempR += hFlipFactorPerChannel;
            srcPtrTempG += hFlipFactorPerChannel;
            srcPtrTempB += hFlipFactorPerChannel;

            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR);
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG);
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB);
                srcPtrTempR += srcPtrIncrementPerPixel;
                srcPtrTempG += srcPtrIncrementPerPixel;
                srcPtrTempB += srcPtrIncrementPerPixel;
            }

            srcPtrRowR += hStrideSrcIncrement;
            srcPtrRowG += hStrideSrcIncrement;
            srcPtrRowB += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[3];

                rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);                       // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);  // simd stores

                srcPtrTemp += srcPtrIncrement;
                dstPtrTemp += vectorIncrement;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
                *dstPtrTemp++ = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = (bufferLength / 8) * 8;
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
                    __m256 p[1];

                    rpp_simd_load(load8Fn, srcPtrTemp, p);                     // simd loads
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);  // simd stores

                    srcPtrTemp += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcPtrTemp += hFlipFactorPerChannel;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = (Rpp16f)RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp);
                    srcPtrTemp += srcPtrIncrementPerPixel;
                    dstPtrTemp++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                   Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType, RppLayoutParams layoutParams,
                                   rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        flip_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                               horizontalTensor[batchCount], verticalTensor[batchCount], roi,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

static inline RppStatus flip_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr,
                                             Rpp8s* dstPtrImage, RpptDescPtr dstDescPtr,
                                             Rpp32u horizontalFlag, Rpp32u verticalFlag,
                                             RpptROI roi, RppLayoutParams layoutParams) {
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp8s *srcPtrChannel, *dstPtrChannel;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = (bufferLength / 48) * 48;
    Rpp32u vectorIncrement = 48;
    Rpp32u vectorIncrementPerChannel = 16;

    // Initialize hFactor, vFactor and hStrideSrcIncrement with default values (used for getting src
    // pointer location)
    Rpp32u hFactor = roi.xywhROI.xy.x * layoutParams.bufferMultiplier;
    Rpp32u vFactor = roi.xywhROI.xy.y * srcDescPtr->strides.hStride;
    Rpp32s hStrideSrcIncrement = srcDescPtr->strides.hStride;

    // Initialize load functions with default values
    auto load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_avx;
    auto load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_avx;
    auto load16Fn = &rpp_load16_i8_to_f32_avx;

    // Update the load functions, hFactor, vFactor and hStrideSrcIncrement based on the flags
    // enabled
    if (horizontalFlag == 1) {
        hFactor +=
            (roi.xywhROI.roiWidth - vectorIncrementPerChannel) * layoutParams.bufferMultiplier;
        load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
        load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
        load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
    }
    if (verticalFlag == 1) {
        vFactor += (roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
        hStrideSrcIncrement = -srcDescPtr->strides.hStride;
    }
    srcPtrChannel = srcPtrImage + vFactor + hFactor;

    // Compute constant increment, Decrement factors used in source pointer updation
    Rpp32s srcPtrIncrement = (horizontalFlag) ? -vectorIncrement : vectorIncrement;
    Rpp32u hFlipFactor = (vectorIncrement - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerChannel =
        (horizontalFlag) ? -vectorIncrementPerChannel : vectorIncrementPerChannel;
    Rpp32u hFlipFactorPerChannel = (vectorIncrementPerChannel - 1) * horizontalFlag;
    Rpp32s srcPtrIncrementPerRGB = (horizontalFlag) ? -3 : 3;
    Rpp32s srcPtrIncrementPerPixel = (horizontalFlag) ? -1 : 1;

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0
    // and verticalflag = 0)
    if (!(horizontalFlag | verticalFlag) && (srcDescPtr->layout == dstDescPtr->layout)) {
        for (int c = 0; c < layoutParams.channelParam; c++) {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                memcpy(dstPtrRow, srcPtrRow, bufferLength);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    // flip with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];

                rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);  // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, p);  // simd stores

                srcPtrTemp += srcPtrIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTempR++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[0]));
                *dstPtrTempG++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[1]));
                *dstPtrTempB++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[2]));
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRowR += dstDescPtr->strides.hStride;
            dstPtrRowG += dstDescPtr->strides.hStride;
            dstPtrRowB += dstDescPtr->strides.hStride;
        }
    }

    // flip with fused output-layout toggle (NCHW -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 p[6];

                rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB,
                              p);                                                  // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);  // simd stores

                srcPtrTempR += srcPtrIncrementPerChannel;
                srcPtrTempG += srcPtrIncrementPerChannel;
                srcPtrTempB += srcPtrIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
            }

            srcPtrTempR += hFlipFactorPerChannel;
            srcPtrTempG += hFlipFactorPerChannel;
            srcPtrTempB += hFlipFactorPerChannel;

            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(*srcPtrTempR));
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(*srcPtrTempG));
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(*srcPtrTempB));
                srcPtrTempR += srcPtrIncrementPerPixel;
                srcPtrTempG += srcPtrIncrementPerPixel;
                srcPtrTempB += srcPtrIncrementPerPixel;
            }

            srcPtrRowR += hStrideSrcIncrement;
            ;
            srcPtrRowG += hStrideSrcIncrement;
            srcPtrRowB += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC)
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
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                __m256 p[6];
                rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);                      // simd loads
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);  // simd stores
                srcPtrTemp += srcPtrIncrement;
                dstPtrTemp += vectorIncrement;
            }
            srcPtrTemp += hFlipFactor;
            for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[0]));
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[1]));
                *dstPtrTemp++ = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(srcPtrTemp[2]));
                srcPtrTemp += srcPtrIncrementPerRGB;
            }

            srcPtrRow += hStrideSrcIncrement;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else {
        Rpp32u alignedLength = (bufferLength / 16) * 16;
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
                    __m256 p[2];
                    rpp_simd_load(load16Fn, srcPtrTemp, p);                    // simd loads
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += srcPtrIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTemp = (Rpp8s)RPPPIXELCHECKI8((Rpp32f)(*srcPtrTemp));
                    srcPtrTemp += srcPtrIncrementPerPixel;
                    dstPtrTemp++;
                }

                srcPtrRow += hStrideSrcIncrement;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus flip_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                 Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType, RppLayoutParams layoutParams,
                                 rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        flip_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                             horizontalTensor[batchCount], verticalTensor[batchCount], roi,
                             layoutParams);
    }

    return RPP_SUCCESS;
}

// -------------------- Single Image Processing --------------------

RppStatus flip_u8_u8_host_single_image(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                       Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType, RppLayoutParams layoutParams,
                                       rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(&roiTensorPtrSrc[0], &roi, &roiDefault, roiType);
    return flip_u8_u8_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, horizontalTensor[0],
                                verticalTensor[0], roi, layoutParams);
}

RppStatus flip_f32_f32_host_single_image(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                         RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                         Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType, RppLayoutParams layoutParams,
                                         rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(&roiTensorPtrSrc[0], &roi, &roiDefault, roiType);
    return flip_f32_f32_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, horizontalTensor[0],
                                  verticalTensor[0], roi, layoutParams);
}

RppStatus flip_f16_f16_host_single_image(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                         RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                         Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType, RppLayoutParams layoutParams,
                                         rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(&roiTensorPtrSrc[0], &roi, &roiDefault, roiType);
    return flip_f16_f16_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, horizontalTensor[0],
                                  verticalTensor[0], roi, layoutParams);
}

RppStatus flip_i8_i8_host_single_image(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                       Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType, RppLayoutParams layoutParams,
                                       rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(&roiTensorPtrSrc[0], &roi, &roiDefault, roiType);
    return flip_i8_i8_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, horizontalTensor[0],
                                verticalTensor[0], roi, layoutParams);
}

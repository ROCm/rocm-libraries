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
#include "rpp_cpu_random.hpp"

inline void compute_jitter_src_loc_avx(__m256i* pxXorwowStateX, __m256i* pxXorwowStateCounter,
                                       __m256& pRow, __m256& pCol, __m256& pKernelSize,
                                       __m256& pBound, __m256& pHeightLimit, __m256& pWidthLimit,
                                       __m256& pStride, __m256& pChannel, Rpp32s* srcLoc) {
    __m256 pRngX = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pRngY = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pX = _mm256_mul_ps(pRngX, pKernelSize);
    __m256 pY = _mm256_mul_ps(pRngY, pKernelSize);
    pX =
        _mm256_max_ps(_mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pRow, _mm256_sub_ps(pX, pBound))),
                                    pHeightLimit),
                      avx_p0);
    pY = _mm256_max_ps(
        _mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pCol, _mm256_sub_ps(pY, pBound))), pWidthLimit),
        avx_p0);
    __m256i pxSrcLoc =
        _mm256_cvtps_epi32(_mm256_fmadd_ps(pX, pStride, _mm256_mul_ps(pY, pChannel)));
    _mm256_storeu_si256((__m256i*)srcLoc, pxSrcLoc);
}

inline void compute_jitter_src_loc(RpptXorwowStateBoxMuller* xorwowState, Rpp32s row, Rpp32s col,
                                   Rpp32s kSize, Rpp32s heightLimit, Rpp32s widthLimit,
                                   Rpp32s stride, Rpp32s bound, Rpp32s channels, Rpp32s& loc) {
    Rpp32u heightIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    Rpp32u widthIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    loc = std::max(std::min(static_cast<int>(row + heightIncrement - bound), heightLimit), 0) *
          stride;
    loc += std::max(std::min(static_cast<int>(col + widthIncrement - bound), (widthLimit - 1)), 0) *
           channels;
}

inline void jitter_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8u* dstPtrImage,
                                   RpptDescPtr dstDescPtr, RpptROI roi, Rpp32u kernelSize,
                                   RpptXorwowStateBoxMuller* xorwowInitialStatePtr, Rpp32u offset,
                                   RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bound = (kernelSize - 1) / 2;
    Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pKernelSize = _mm256_set1_ps(kernelSize);
    __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
    __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
    __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth - 1);
    __m256 pBound = _mm256_set1_ps(bound);

    // Jitter with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8u* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pkd3_avx(srcPtrChannel, srcLocArray, pxRow);
                rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                *dstPtrTempR++ = *(srcPtrChannel + loc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
            }
        }
    }

    // Jitter with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            // Per-thread source pointers (NCHW layout has planar channels)
            Rpp8u* srcPtrRowR = srcPtrChannel;
            Rpp8u* srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            Rpp8u* srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: rpp_store24_u8pln3_to_u8pkd3_avx writes 32 bytes but we only
            // need 24
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow[3];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowR, srcLocArray, pxRow[0]);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowG, srcLocArray, pxRow[1]);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowB, srcLocArray, pxRow[2]);
                rpp_simd_store(rpp_store24_u8pln3_to_u8pkd3_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRowR + loc);
                *dstPtrTemp++ = *(srcPtrRowG + loc);
                *dstPtrTemp++ = *(srcPtrRowB + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            // Per-thread source pointer (NHWC layout has interleaved channels)
            Rpp8u* srcPtrRow = srcPtrChannel;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: rpp_store24_u8_to_u8_avx writes 32 bytes but we only need 24
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pkd3_avx(srcPtrRow, srcLocArray, pxRow);
                rpp_simd_store(rpp_store24_u8_to_u8_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRow + loc);
                *dstPtrTemp++ = *(srcPtrRow + 1 + loc);
                *dstPtrTemp++ = *(srcPtrRow + 2 + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8u* dstPtrTempChn = dstPtrTemp;
                Rpp8u* srcPtrTempChn = srcPtrChannel;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    __m256i pxRow;
                    rpp_resize_nn_extract_pln1_avx(srcPtrTempChn, srcLocArray, pxRow);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChn), _mm256_castsi256_si128(pxRow));
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp8u* dstPtrTempChn = dstPtrTemp;
                Rpp8u* srcPtrTempChn = srcPtrChannel;
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    *dstPtrTempChn = *(srcPtrTempChn + loc);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += layoutParams.bufferMultiplier;
            }
        }
    }
}

RppStatus jitter_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32u* kernelSizeTensor,
                                   RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        // RNG offset must be in pixel count (2 RNG calls per pixel), not bytes
        Rpp32u offset = batchCount * roi.xywhROI.roiHeight * roi.xywhROI.roiWidth * 2;

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        jitter_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, kernelSize,
                               xorwowInitialStatePtr, offset, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void jitter_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                     Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr, RpptROI roi,
                                     Rpp32u kernelSize,
                                     RpptXorwowStateBoxMuller* xorwowInitialStatePtr, Rpp32u offset,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bound = (kernelSize - 1) / 2;
    Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pKernelSize = _mm256_set1_ps(kernelSize);
    __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
    __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
    __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth - 1);
    __m256 pBound = _mm256_set1_ps(bound);

    // Jitter with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp32f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pxRow[3];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3_avx, srcPtrChannel, srcLocArray,
                              pxRow);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTempR++ = *(srcPtrChannel + loc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
            }
        }
    }

    // Jitter with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: rpp_store24_f32pln3_to_f32pkd3_avx may write past buffer
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pxRow[4];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowR, srcLocArray, pxRow[0]);
                rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowG, srcLocArray, pxRow[1]);
                rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowB, srcLocArray, pxRow[2]);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRowR + loc);
                *dstPtrTemp++ = *(srcPtrRowG + loc);
                *dstPtrTemp++ = *(srcPtrRowB + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* srcPtrRow = srcPtrChannel;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                // Manual gather for F32 PKD3 -> PKD3
                for (int i = 0; i < vectorIncrementPerChannel; i++) {
                    Rpp32s loc = srcLocArray[i];
                    *dstPtrTemp++ = *(srcPtrRow + loc);
                    *dstPtrTemp++ = *(srcPtrRow + loc + 1);
                    *dstPtrTemp++ = *(srcPtrRow + loc + 2);
                }
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRow + loc);
                *dstPtrTemp++ = *(srcPtrRow + loc + 1);
                *dstPtrTemp++ = *(srcPtrRow + loc + 2);
            }
        }
    }

    // Jitter without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp32f* srcPtrTempChn = srcPtrChannel;
                Rpp32f* dstPtrTempChn = dstPtrTemp;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    __m256 pxRow;
                    rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrTempChn, srcLocArray,
                                  pxRow);
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTempChn, &pxRow);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32f* dstPtrTempChn = dstPtrTemp;
                Rpp32f* srcPtrTempChn = srcPtrChannel;
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    *dstPtrTempChn = *(srcPtrTempChn + loc);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus jitter_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u* kernelSizeTensor,
                                     RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        // RNG offset must be in pixel count (2 RNG calls per pixel), not bytes
        Rpp32u offset = batchCount * roi.xywhROI.roiHeight * roi.xywhROI.roiWidth * 2;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        jitter_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, kernelSize,
                                 xorwowInitialStatePtr, offset, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void jitter_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                     Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr, RpptROI roi,
                                     Rpp32u kernelSize,
                                     RpptXorwowStateBoxMuller* xorwowInitialStatePtr, Rpp32u offset,
                                     RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bound = (kernelSize - 1) / 2;
    Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pKernelSize = _mm256_set1_ps(kernelSize);
    __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
    __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
    __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth - 1);
    __m256 pBound = _mm256_set1_ps(bound);

    // Jitter with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp16f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pxRow[3];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_simd_load(rpp_resize_nn_load_f16pkd3_to_f32pln3_avx, srcPtrChannel, srcLocArray,
                              pxRow);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTempR++ = *(srcPtrChannel + loc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
            }
        }
    }

    // Jitter with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: store may write past buffer
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pxRow[4];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_load_f16pln1_avx(srcPtrRowR, srcLocArray, pxRow[0]);
                rpp_resize_nn_load_f16pln1_avx(srcPtrRowG, srcLocArray, pxRow[1]);
                rpp_resize_nn_load_f16pln1_avx(srcPtrRowB, srcLocArray, pxRow[2]);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRowR + loc);
                *dstPtrTemp++ = *(srcPtrRowG + loc);
                *dstPtrTemp++ = *(srcPtrRowB + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* srcPtrRow = srcPtrChannel;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                // Manual gather for F16 PKD3 -> PKD3
                for (int i = 0; i < vectorIncrementPerChannel; i++) {
                    Rpp32s loc = srcLocArray[i];
                    *dstPtrTemp++ = *(srcPtrRow + loc);
                    *dstPtrTemp++ = *(srcPtrRow + loc + 1);
                    *dstPtrTemp++ = *(srcPtrRow + loc + 2);
                }
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRow + loc);
                *dstPtrTemp++ = *(srcPtrRow + loc + 1);
                *dstPtrTemp++ = *(srcPtrRow + loc + 2);
            }
        }
    }

    // Jitter without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp16f* srcPtrTempChn = srcPtrChannel;
                Rpp16f* dstPtrTempChn = dstPtrTemp;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    __m256 pxRow;
                    rpp_resize_nn_load_f16pln1_avx(srcPtrTempChn, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTempChn, &pxRow);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp16f* dstPtrTempChn = dstPtrTemp;
                Rpp16f* srcPtrTempChn = srcPtrChannel;
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    *dstPtrTempChn = *(srcPtrTempChn + loc);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus jitter_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u* kernelSizeTensor,
                                     RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        // RNG offset must be in pixel count (2 RNG calls per pixel), not bytes
        Rpp32u offset = batchCount * roi.xywhROI.roiHeight * roi.xywhROI.roiWidth * 2;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        jitter_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, kernelSize,
                                 xorwowInitialStatePtr, offset, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void jitter_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8s* dstPtrImage,
                                   RpptDescPtr dstDescPtr, RpptROI roi, Rpp32u kernelSize,
                                   RpptXorwowStateBoxMuller* xorwowInitialStatePtr, Rpp32u offset,
                                   RppLayoutParams layoutParams, Rpp32u intraThreads) {
    Rpp32u bound = (kernelSize - 1) / 2;
    Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;

    __m256 pKernelSize = _mm256_set1_ps(kernelSize);
    __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
    __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
    __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth - 1);
    __m256 pBound = _mm256_set1_ps(bound);

    // Jitter with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8s* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pkd3_avx(srcPtrChannel, srcLocArray, pxRow);
                rpp_simd_store(rpp_store24_i8pkd3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                *dstPtrTempR++ = *(srcPtrChannel + loc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
            }
        }
    }

    // Jitter with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: rpp_store24_i8pln3_to_i8pkd3_avx may write past buffer
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow[3];
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowR, srcLocArray, pxRow[0]);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowG, srcLocArray, pxRow[1]);
                rpp_resize_nn_extract_pln1_avx(srcPtrRowB, srcLocArray, pxRow[2]);
                rpp_simd_store(rpp_store24_i8pln3_to_i8pkd3_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRowR + loc);
                *dstPtrTemp++ = *(srcPtrRowG + loc);
                *dstPtrTemp++ = *(srcPtrRowB + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8s* srcPtrRow = srcPtrChannel;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            // Buffer overrun guard: rpp_store24_i8_to_i8_avx writes 32 bytes but we only need 24
            Rpp32u alignedLengthPkd = (alignedLength >= vectorIncrementPerChannel)
                                          ? (alignedLength - vectorIncrementPerChannel)
                                          : 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256i pxRow;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                rpp_resize_nn_extract_pkd3_avx(srcPtrRow, srcLocArray, pxRow);
                rpp_simd_store(rpp_store24_i8_to_i8_avx, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                *dstPtrTemp++ = *(srcPtrRow + loc);
                *dstPtrTemp++ = *(srcPtrRow + 1 + loc);
                *dstPtrTemp++ = *(srcPtrRow + 2 + loc);
            }
        }
    }

    // Jitter without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            RpptXorwowStateBoxMuller xorwowState;
            __m256i pxXorwowStateX[5], pxXorwowStateCounter;
            Rpp32s srcLocArray[8] = {0};
            Rpp32u rowOffset = offset + dstLocRow * roi.xywhROI.roiWidth * 2;

            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            __m256 pRow = _mm256_set1_ps(dstLocRow);
            __m256 pCol = avx_pDstLocInit;

            // Initialize RNG state (both scalar and AVX)
            rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, rowOffset,
                                                    pxXorwowStateX, &pxXorwowStateCounter);

            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8s* dstPtrTempChn = dstPtrTemp;
                Rpp8s* srcPtrTempChn = srcPtrChannel;
                compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol,
                                           pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride,
                                           pChannel, srcLocArray);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    __m256i pxRow;
                    rpp_resize_nn_extract_pln1_avx(srcPtrTempChn, srcLocArray, pxRow);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChn), _mm256_castsi256_si128(pxRow));
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                pCol = _mm256_add_ps(avx_p8, pCol);
            }
#endif
            // Initialize scalar RNG state for tail loop, offset by pixels already processed
            Rpp32u scalarOffset = rowOffset + vectorLoopCount * 2;
            xorwowState = xorwowInitialStatePtr[0];
            xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + scalarOffset;

            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp8s* dstPtrTempChn = dstPtrTemp;
                Rpp8s* srcPtrTempChn = srcPtrChannel;
                Rpp32s loc;
                compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize,
                                       heightLimit, roi.xywhROI.roiWidth,
                                       srcDescPtr->strides.hStride, bound,
                                       layoutParams.bufferMultiplier, loc);
                for (int c = 0; c < srcDescPtr->c; c++) {
                    *dstPtrTempChn = *(srcPtrTempChn + loc);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += layoutParams.bufferMultiplier;
            }
        }
    }
}

RppStatus jitter_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32u* kernelSizeTensor,
                                   RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        // RNG offset must be in pixel count (2 RNG calls per pixel), not bytes
        Rpp32u offset = batchCount * roi.xywhROI.roiHeight * roi.xywhROI.roiWidth * 2;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        jitter_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, roi, kernelSize,
                               xorwowInitialStatePtr, offset, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

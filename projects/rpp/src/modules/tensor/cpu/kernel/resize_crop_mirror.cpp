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
#include "rpp_cpu_geometric.hpp"
#include "rpp_cpu_interpolation.hpp"

inline void clamp_src_locations(Rpp32s* locArray, int count, int maxWidthLimit) {
    for (int i = 0; i < count; i++) locArray[i] = std::clamp(locArray[i], 0, maxWidthLimit);
}

// U8 impl function
inline void resize_crop_mirror_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr,
                                               Rpp8u* dstPtrImage, RpptDescPtr dstDescPtr,
                                               RpptImagePatch dstImgSize, Rpp32u mirrorFlag,
                                               RpptROI roi, RppLayoutParams layoutParams,
                                               Rpp32u intraThreads) {
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize.width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize.height));
    Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 2;
    Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 2) * srcDescPtr->strides.wStride;
    Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
    Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrementPkd = 24;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = dstImgSize.width & ~7;
    // For packed output (NHWC), reduce alignedLength to avoid SIMD store overflow at row boundaries
    // The 32-byte AVX2 store writes 8 extra bytes beyond 24 bytes needed, so leave last 8 pixels
    // for scalar loop
    Rpp32u alignedLengthPkd = (alignedLength > 8) ? (alignedLength - 8) : 0;
    __m256 pWRatio = _mm256_set1_ps(wRatio);
    __m256 pWOffset = _mm256_set1_ps(wOffset);
    __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
    Rpp32u width = dstImgSize.width;

    __m256 pDstLocInit = avx_pDstLocInit;
    auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
    if (mirrorFlag) {
        pDstLocInit = _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5,
                                     width - 6, width - 7, width - 8);
        computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
    }

    // Resize Crop Mirror with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8u* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, dstPtrTempR,
                                                      dstPtrTempG, dstPtrTempB);
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Resize Crop Mirror with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8u* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8u* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0],
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2],
                              srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4],
                              srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8u* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8u* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8u* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            Rpp8u* dstPtrTempChn;
            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                dstPtrTempChn = dstPtrTemp;
                __m256 pSrc[4], pDst;
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx,
                                  &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE],
                                  srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                                  maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                dstPtrTempChn = dstPtrTemp;
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    compute_bilinear_interpolation_1c(
                        &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn,
                        maxWidthLimit, bilinearCoeffs, dstPtrTempChn);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

// U8 tensor function
RppStatus resize_crop_mirror_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                               RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSize,
                                               Rpp32u* mirrorTensor, RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType, RppLayoutParams layoutParams,
                                               rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads =
            get_intra_image_threads(handle, dstDescPtr->n, dstImgSize[batchCount].height);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        resize_crop_mirror_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                           dstImgSize[batchCount], mirrorTensor[batchCount], roi,
                                           layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// F32 impl function
inline void resize_crop_mirror_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                                 Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                                 RpptImagePatch dstImgSize, Rpp32u mirrorFlag,
                                                 RpptROI roi, RppLayoutParams layoutParams,
                                                 Rpp32u intraThreads) {
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize.width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize.height));
    Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 2;
    Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 2) * srcDescPtr->strides.wStride;
    Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
    Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrementPkd = 24;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = dstImgSize.width & ~7;
    // For packed output (NHWC), reduce alignedLength to avoid SIMD store overflow at row boundaries
    // The 32-byte AVX2 store writes 8 extra bytes beyond 24 bytes needed, so leave last 8 pixels
    // for scalar loop
    Rpp32u alignedLengthPkd = (alignedLength > 8) ? (alignedLength - 8) : 0;
    __m256 pWRatio = _mm256_set1_ps(wRatio);
    __m256 pWOffset = _mm256_set1_ps(wOffset);
    __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
    Rpp32u width = dstImgSize.width;

    __m256 pDstLocInit = avx_pDstLocInit;
    auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
    if (mirrorFlag) {
        pDstLocInit = _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5,
                                     width - 6, width - 7, width - 8);
        computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
    }

    // Resize Crop Mirror with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, dstPtrTempR,
                                                      dstPtrTempG, dstPtrTempB);
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Resize Crop Mirror with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp32f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0],
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2],
                              srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4],
                              srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp32f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp32f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            Rpp32f* dstPtrTempChn;
            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                dstPtrTempChn = dstPtrTemp;
                __m256 pSrc[4], pDst;
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx,
                                  &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE],
                                  srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                                  maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                dstPtrTempChn = dstPtrTemp;
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    compute_bilinear_interpolation_1c(
                        &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn,
                        maxWidthLimit, bilinearCoeffs, dstPtrTempChn);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

// F32 tensor function
RppStatus resize_crop_mirror_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr,
                                                 Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize, Rpp32u* mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads =
            get_intra_image_threads(handle, dstDescPtr->n, dstImgSize[batchCount].height);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);

        Rpp32f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        resize_crop_mirror_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                             dstImgSize[batchCount], mirrorTensor[batchCount], roi,
                                             layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// F16 impl function
inline void resize_crop_mirror_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                                 Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                                 RpptImagePatch dstImgSize, Rpp32u mirrorFlag,
                                                 RpptROI roi, RppLayoutParams layoutParams,
                                                 Rpp32u intraThreads) {
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize.width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize.height));
    Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 2;
    Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 2) * srcDescPtr->strides.wStride;
    Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
    Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrementPkd = 24;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = dstImgSize.width & ~7;
    // F16 uses safe partial stores (rpp_storeu_si64, rpp_storeu_si32), no overflow guard needed
    __m256 pWRatio = _mm256_set1_ps(wRatio);
    __m256 pWOffset = _mm256_set1_ps(wOffset);
    __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
    Rpp32u width = dstImgSize.width;

    __m256 pDstLocInit = avx_pDstLocInit;
    auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
    if (mirrorFlag) {
        pDstLocInit = _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5,
                                     width - 6, width - 7, width - 8);
        computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
    }

    // Resize Crop Mirror with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp16f* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, dstPtrTempR,
                                                      dstPtrTempG, dstPtrTempB);
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Resize Crop Mirror with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp16f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp16f* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0],
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2],
                              srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4],
                              srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp16f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp16f* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp16f* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            Rpp16f* dstPtrTempChn;
            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                dstPtrTempChn = dstPtrTemp;
                __m256 pSrc[4], pDst;
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx,
                                  &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE],
                                  srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                                  maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                dstPtrTempChn = dstPtrTemp;
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    compute_bilinear_interpolation_1c(
                        &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn,
                        maxWidthLimit, bilinearCoeffs, dstPtrTempChn);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

// F16 tensor function
RppStatus resize_crop_mirror_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr,
                                                 Rpp16f* dstPtr, RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize, Rpp32u* mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads =
            get_intra_image_threads(handle, dstDescPtr->n, dstImgSize[batchCount].height);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);

        Rpp16f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        resize_crop_mirror_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                             dstImgSize[batchCount], mirrorTensor[batchCount], roi,
                                             layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// I8 impl function
inline void resize_crop_mirror_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr,
                                               Rpp8s* dstPtrImage, RpptDescPtr dstDescPtr,
                                               RpptImagePatch dstImgSize, Rpp32u mirrorFlag,
                                               RpptROI roi, RppLayoutParams layoutParams,
                                               Rpp32u intraThreads) {
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize.width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize.height));
    Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 2;
    Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 2) * srcDescPtr->strides.wStride;
    Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
    Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrementPkd = 24;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = dstImgSize.width & ~7;
    // For packed output (NHWC), reduce alignedLength to avoid SIMD store overflow at row boundaries
    // The 32-byte AVX2 store writes 8 extra bytes beyond 24 bytes needed, so leave last 8 pixels
    // for scalar loop
    Rpp32u alignedLengthPkd = (alignedLength > 8) ? (alignedLength - 8) : 0;
    __m256 pWRatio = _mm256_set1_ps(wRatio);
    __m256 pWOffset = _mm256_set1_ps(wOffset);
    __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
    Rpp32u width = dstImgSize.width;

    __m256 pDstLocInit = avx_pDstLocInit;
    auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
    if (mirrorFlag) {
        pDstLocInit = _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5,
                                     width - 6, width - 7, width - 8);
        computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
    }

    // Resize Crop Mirror with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8s* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, dstPtrTempR,
                                                      dstPtrTempG, dstPtrTempB);
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }
        }
    }

    // Resize Crop Mirror with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8s* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8s* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0],
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2],
                              srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4],
                              srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8s* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp8s* srcRowPtrsForInterp[2];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLengthPkd;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrc[12], pDst[3];
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, true);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);
                rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp,
                              srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                              maxWidthLimitMinusStride);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset,
                                                            srcDescPtr->strides.wStride);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn,
                                                      maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0],
                                                      &dstPtrTemp[1], &dstPtrTemp[2]);
                dstPtrTemp += dstDescPtr->c;
            }
        }
    }

    // Resize Crop Mirror without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRow;
        dstPtrRow = dstPtrChannel;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < dstImgSize.height; dstLocRow++) {
            // Thread-local variables to avoid data races
            __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS],
                pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
            Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS],
                bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
            Rpp32s srcLocationColumnArray[8] = {0};
            Rpp32s srcLocationRow, srcLocationColumn;

            Rpp8s* dstPtrTemp;
            dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* srcRowPtrsForInterp[6];
            compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow,
                                                        &weightParams[0], hOffset);
            compute_src_row_ptrs_for_bilinear_interpolation_pln(
                srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr);
            pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
            pWeightParams[1] = _mm256_set1_ps(weightParams[1]);
            pDstLoc = pDstLocInit;

            Rpp8s* dstPtrTempChn;
            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                dstPtrTempChn = dstPtrTemp;
                __m256 pSrc[4], pDst;
                __m256i pxSrcLoc;
                computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2],
                                   pxSrcLoc, pWOffset, false);
                compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                clamp_src_locations(srcLocationColumnArray, 8, maxWidthLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx,
                                  &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE],
                                  srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc,
                                  maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < dstImgSize.width; vectorLoopCount++) {
                dstPtrTempChn = dstPtrTemp;
                int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn,
                                                            &weightParams[2], wOffset);
                compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    compute_bilinear_interpolation_1c(
                        &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn,
                        maxWidthLimit, bilinearCoeffs, dstPtrTempChn);
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

// I8 tensor function
RppStatus resize_crop_mirror_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                               RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSize,
                                               Rpp32u* mirrorTensor, RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType, RppLayoutParams layoutParams,
                                               rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u intraThreads =
            get_intra_image_threads(handle, dstDescPtr->n, dstImgSize[batchCount].height);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);

        Rpp8s* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        resize_crop_mirror_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                           dstImgSize[batchCount], mirrorTensor[batchCount], roi,
                                           layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

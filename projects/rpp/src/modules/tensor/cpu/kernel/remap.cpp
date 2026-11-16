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

inline void compute_remap_src_loc_sse(Rpp32f* rowRemapTablePtr, Rpp32f* colRemapTablePtr,
                                      Rpp32s* locArray, __m128& pStride, __m128& pWidthLimit,
                                      __m128& pHeightLimit, const __m128& pChannel = xmm_p1) {
    __m128 pRowRemapVal = _mm_loadu_ps(rowRemapTablePtr);
    pRowRemapVal = _mm_max_ps(_mm_min_ps(pRowRemapVal, pHeightLimit), xmm_p0);
    __m128 pColRemapVal = _mm_loadu_ps(colRemapTablePtr);
    pColRemapVal = _mm_max_ps(_mm_min_ps(pColRemapVal, pWidthLimit), xmm_p0);
    __m128i pxRemappedSrcLoc =
        _mm_cvtps_epi32(_mm_fmadd_ps(pRowRemapVal, pStride, _mm_mul_ps(pColRemapVal, pChannel)));
    _mm_storeu_si128((__m128i*)locArray, pxRemappedSrcLoc);
}

inline void compute_remap_src_loc(Rpp32f rowLoc, Rpp32f colLoc, Rpp32s& srcLoc, Rpp32s stride,
                                  Rpp32f widthLimit, Rpp32f heightLimit, Rpp32s channels = 1) {
    rowLoc = std::max(0.0f, std::min(rowLoc, heightLimit));
    colLoc = std::max(0.0f, std::min(colLoc, widthLimit));
    srcLoc = (rowLoc * stride) + colLoc * channels;
}

/************* NEAREST NEIGHBOR INTERPOLATION *************/

inline void remap_nn_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8u* dstPtrImage,
                                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTableImage,
                                     Rpp32f* colRemapTableImage, RpptDescPtr remapTableDescPtr,
                                     RpptROI roi, Rpp32u intraThreads) {
    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrement = 12;
    Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);
    __m128 pWidthLimit = _mm_set1_ps(widthLimit);
    __m128 pHeightLimit = _mm_set1_ps(heightLimit);

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, remapSrcLocArrayLocal,
                              pxRow);
                rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        Rpp8u* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: SIMD store writes 16 bytes but only 12 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 4
        Rpp32u alignedLengthPKD = (alignedLength >= 4) ? (alignedLength - 4) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow[3];
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);
                rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowR, remapSrcLocArrayLocal,
                              pxRow[0]);
                rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowG, remapSrcLocArrayLocal,
                              pxRow[1]);
                rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowB, remapSrcLocArrayLocal,
                              pxRow[2]);
                rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: SIMD store writes 16 bytes but only 12 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 4
        Rpp32u alignedLengthPKD = (alignedLength >= 4) ? (alignedLength - 4) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, remapSrcLocArrayLocal,
                              pxRow);
                rpp_simd_store(rpp_store12_u8_to_u8, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8u *srcPtrTempChn, *dstPtrTempChn;
                srcPtrTempChn = srcPtrChannel;
                dstPtrTempChn = dstPtrTemp;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    __m128i pxRow;
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, remapSrcLocArrayLocal,
                                  pxRow);
                    rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pxRow);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp8u* dstPtrTempChannel = dstPtrTemp;
                Rpp8u* srcPtrTempChannel = srcPtrChannel;
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                    dstPtrTempChannel += dstDescPtr->strides.cStride;
                    srcPtrTempChannel += srcDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus remap_nn_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                     Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
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

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_nn_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 rowRemapTableImage, colRemapTableImage, remapTableDescPtr, roi,
                                 intraThreads);
    }

    return RPP_SUCCESS;
}
inline void remap_nn_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                       Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                       RpptDescPtr remapTableDescPtr, RpptROI roi,
                                       Rpp32u intraThreads) {
    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrement = 12;
    Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);
    __m128 pWidthLimit = _mm_set1_ps(widthLimit);
    __m128 pHeightLimit = _mm_set1_ps(heightLimit);

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128 pRow[3];
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrChannel,
                              remapSrcLocArrayLocal, pRow);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        Rpp32f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: SIMD store writes 16 bytes but only 12 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 4
        Rpp32u alignedLengthPKD = (alignedLength >= 4) ? (alignedLength - 4) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m128 pRow[4];
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);
                rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowR, remapSrcLocArrayLocal,
                              pRow[0]);
                rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowG, remapSrcLocArrayLocal,
                              pRow[1]);
                rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowB, remapSrcLocArrayLocal,
                              pRow[2]);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust for PKD3 output: store4 writes 16 bytes but only 12 are valid (1 pixel at a time)
        // Need to ensure last SIMD write doesn't overflow into next row (1 pixel = 3 F32s)
        // Last safe pixel for SSE: W-2, so loop limit is W-1 (since loop uses <)
        Rpp32u alignedLengthPKD = (roi.xywhROI.roiWidth >= 2) ? (roi.xywhROI.roiWidth - 1) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD; vectorLoopCount++) {
                __m128 pRow;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrChannel + *remapSrcLocArrayLocal),
                              &pRow);
                rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &pRow);
                dstPtrTemp += 3;
                rowRemapTableTemp++;
                colRemapTableTemp++;
            }
            // Scalar loop for remaining pixels
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                srcPtrTempChn = srcPtrChannel;
                dstPtrTempChn = dstPtrTemp;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    __m128 pRow;
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempChn, remapSrcLocArrayLocal,
                                  pRow);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp32f* dstPtrTempChannel = dstPtrTemp;
                Rpp32f* srcPtrTempChannel = srcPtrChannel;
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                    dstPtrTempChannel += dstDescPtr->strides.cStride;
                    srcPtrTempChannel += srcDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus remap_nn_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                       Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
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

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_nn_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                   rowRemapTableImage, colRemapTableImage, remapTableDescPtr, roi,
                                   intraThreads);
    }

    return RPP_SUCCESS;
}
inline void remap_nn_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr, Rpp8s* dstPtrImage,
                                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTableImage,
                                     Rpp32f* colRemapTableImage, RpptDescPtr remapTableDescPtr,
                                     RpptROI roi, Rpp32u intraThreads) {
    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage;
    dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrement = 12;
    Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);
    __m128 pWidthLimit = _mm_set1_ps(widthLimit);
    __m128 pHeightLimit = _mm_set1_ps(heightLimit);

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrChannel, remapSrcLocArrayLocal,
                              pxRow);
                rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB,
                               pxRow);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        Rpp8s* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: SIMD store writes 16 bytes but only 12 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 4
        Rpp32u alignedLengthPKD = (alignedLength >= 4) ? (alignedLength - 4) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow[3];
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);
                rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowR, remapSrcLocArrayLocal,
                              pxRow[0]);
                rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowG, remapSrcLocArrayLocal,
                              pxRow[1]);
                rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowB, remapSrcLocArrayLocal,
                              pxRow[2]);
                rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: SIMD store writes 16 bytes but only 12 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 4
        Rpp32u alignedLengthPKD = (alignedLength >= 4) ? (alignedLength - 4) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m128i pxRow;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit, pSrcChannel);
                rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrChannel, remapSrcLocArrayLocal,
                              pxRow);
                rpp_simd_store(rpp_store12_i8_to_i8, dstPtrTemp, pxRow);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            Rpp32s remapSrcLocArrayLocal[4] = {0};
            Rpp32s remappedSrcLoc;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                Rpp8s *srcPtrTempChn, *dstPtrTempChn;
                srcPtrTempChn = srcPtrChannel;
                dstPtrTempChn = dstPtrTemp;
                compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp,
                                          remapSrcLocArrayLocal, pSrcStride, pWidthLimit,
                                          pHeightLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    __m128i pxRow;
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempChn, remapSrcLocArrayLocal,
                                  pxRow);
                    rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pxRow);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn += dstDescPtr->strides.cStride;
                }
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                Rpp8s* dstPtrTempChannel = dstPtrTemp;
                Rpp8s* srcPtrTempChannel = srcPtrChannel;
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                for (int c = 0; c < dstDescPtr->c; c++) {
                    *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                    dstPtrTempChannel += dstDescPtr->strides.cStride;
                    srcPtrTempChannel += srcDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus remap_nn_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                     Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
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

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_nn_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                 rowRemapTableImage, colRemapTableImage, remapTableDescPtr, roi,
                                 intraThreads);
    }

    return RPP_SUCCESS;
}
// NN remap impl function with intra-image multithreading (F16 - scalar only)
inline void remap_nn_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                       Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                       Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                       RpptDescPtr remapTableDescPtr, RpptROI roi,
                                       Rpp32u intraThreads) {
    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage;
    dstPtrChannel = dstPtrImage;

    Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s remappedSrcLoc;
            Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        Rpp16f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s remappedSrcLoc;
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);
                *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s remappedSrcLoc;
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit,
                                      srcDescPtr->c);
                *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRow = dstPtrChannel;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32s remappedSrcLoc;
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
            rowRemapTableTemp = rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            colRemapTableTemp = colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            for (int vectorLoopCount = 0; vectorLoopCount < roi.xywhROI.roiWidth;
                 vectorLoopCount++) {
                Rpp16f* dstPtrTempChannel = dstPtrTemp;
                Rpp16f* srcPtrTempChannel = srcPtrChannel;
                compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc,
                                      srcDescPtr->strides.hStride, widthLimit, heightLimit);

                for (int c = 0; c < dstDescPtr->c; c++) {
                    *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                    dstPtrTempChannel += dstDescPtr->strides.cStride;
                    srcPtrTempChannel += srcDescPtr->strides.cStride;
                }
                dstPtrTemp++;
            }
        }
    }
}

RppStatus remap_nn_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                       Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
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

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_nn_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                   rowRemapTableImage, colRemapTableImage, remapTableDescPtr, roi,
                                   intraThreads);
    }

    return RPP_SUCCESS;
}
inline void remap_bilinear_u8_u8_host_impl(Rpp8u* srcPtrImage, RpptDescPtr srcDescPtr,
                                           Rpp8u* dstPtrImage, RpptDescPtr dstDescPtr,
                                           Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                           RpptDescPtr remapTableDescPtr, RpptROI roi,
                                           RpptROI roiLTRB, Rpp32u intraThreads) {
    Rpp8u* srcPtrChannel = srcPtrImage;
    Rpp8u* dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrement = 24;

#if __AVX2__
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
    __m256 pRoiLTRB[4];
    pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
    pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
    pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
    pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRowR = dstPtrChannel;
        Rpp8u* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp8u* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pkd3_to_pln3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: AVX2 store writes 32 bytes but only 24 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 8
        Rpp32u alignedLengthPKD = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: AVX2 store writes 32 bytes but only 24 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 8
        Rpp32u alignedLengthPKD = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRowR = dstPtrChannel;
        Rpp8u* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp8u* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8u* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }

    // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8u* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8u* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[4], pDst;
                compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8u>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }
}
RppStatus remap_bilinear_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                           Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8u* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* rowRemapTableImage =
            rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32f* colRemapTableImage =
            colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_bilinear_u8_u8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                       rowRemapTableImage, colRemapTableImage, remapTableDescPtr,
                                       roi, roiLTRB, intraThreads);
    }

    return RPP_SUCCESS;
}

inline void remap_bilinear_f32_f32_host_impl(Rpp32f* srcPtrImage, RpptDescPtr srcDescPtr,
                                             Rpp32f* dstPtrImage, RpptDescPtr dstDescPtr,
                                             Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                             RpptDescPtr remapTableDescPtr, RpptROI roi,
                                             RpptROI roiLTRB, Rpp32u intraThreads) {
    Rpp32f* srcPtrChannel = srcPtrImage;
    Rpp32f* dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrement = 24;

#if __AVX2__
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
    __m256 pRoiLTRB[4];
    pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
    pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
    pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
    pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRowR = dstPtrChannel;
        Rpp32f* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp32f* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pkd3_to_pln3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRowR = dstPtrChannel;
        Rpp32f* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp32f* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }

    // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp32f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp32f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[4], pDst;
                compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp32f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }
}
RppStatus remap_bilinear_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                             RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                             Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                             RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp32f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* rowRemapTableImage =
            rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32f* colRemapTableImage =
            colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_bilinear_f32_f32_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                         rowRemapTableImage, colRemapTableImage, remapTableDescPtr,
                                         roi, roiLTRB, intraThreads);
    }

    return RPP_SUCCESS;
}
inline void remap_bilinear_i8_i8_host_impl(Rpp8s* srcPtrImage, RpptDescPtr srcDescPtr,
                                           Rpp8s* dstPtrImage, RpptDescPtr dstDescPtr,
                                           Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                           RpptDescPtr remapTableDescPtr, RpptROI roi,
                                           RpptROI roiLTRB, Rpp32u intraThreads) {
    Rpp8s* srcPtrChannel = srcPtrImage;
    Rpp8s* dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrement = 24;

#if __AVX2__
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
    __m256 pRoiLTRB[4];
    pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
    pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
    pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
    pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRowR = dstPtrChannel;
        Rpp8s* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp8s* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pkd3_to_pln3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: AVX2 store writes 32 bytes but only 24 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 8
        Rpp32u alignedLengthPKD = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

        // Adjust alignedLength for PKD3 output: AVX2 store writes 32 bytes but only 24 are valid
        // Need to ensure last SIMD write doesn't overflow into next row
        // Derived from alignedLength: alignedLengthPKD = alignedLength - 8
        Rpp32u alignedLengthPKD = (alignedLength >= 8) ? (alignedLength - 8) : 0;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLengthPKD;
                 vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRowR = dstPtrChannel;
        Rpp8s* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp8s* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp8s* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }

    // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp8s* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp8s* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[4], pDst;
                compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8s>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }
}
RppStatus remap_bilinear_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                           Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8s* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* rowRemapTableImage =
            rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32f* colRemapTableImage =
            colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_bilinear_i8_i8_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                       rowRemapTableImage, colRemapTableImage, remapTableDescPtr,
                                       roi, roiLTRB, intraThreads);
    }

    return RPP_SUCCESS;
}
inline void remap_bilinear_f16_f16_host_impl(Rpp16f* srcPtrImage, RpptDescPtr srcDescPtr,
                                             Rpp16f* dstPtrImage, RpptDescPtr dstDescPtr,
                                             Rpp32f* rowRemapTableImage, Rpp32f* colRemapTableImage,
                                             RpptDescPtr remapTableDescPtr, RpptROI roi,
                                             RpptROI roiLTRB, Rpp32u intraThreads) {
    Rpp16f* srcPtrChannel = srcPtrImage;
    Rpp16f* dstPtrChannel = dstPtrImage;

    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
    Rpp32s vectorIncrementPerChannel = 8;
    Rpp32s vectorIncrement = 24;

#if __AVX2__
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
    __m256 pRoiLTRB[4];
    pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
    pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
    pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
    pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

    // Remap with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
        (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRowR = dstPtrChannel;
        Rpp16f* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp16f* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp16f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pkd3_to_pln3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
        }
    }

    // Remap with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NHWC -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
             (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp16f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, true);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrement;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++) {
                compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel,
                    srcDescPtr);
                dstPtrTemp += 3;
            }
        }
    }

    // Remap without fused output-layout toggle (NCHW -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRowR = dstPtrChannel;
        Rpp16f* dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        Rpp16f* dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp16f* dstPtrTempR = dstPtrRowR + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempG = dstPtrRowG + dstLocRow * dstDescPtr->strides.hStride;
            Rpp16f* dstPtrTempB = dstPtrRowB + dstLocRow * dstDescPtr->strides.hStride;

            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[12], pDst[3];
                compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c,
                                                        pRoiLTRB, false);
                rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                               dstPtrTempB, pDst);
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }

    // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
    else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) &&
             (dstDescPtr->layout == RpptLayout::NCHW)) {
        Rpp16f* dstPtrRow = dstPtrChannel;
        Rpp32f* rowRemapTableRow = rowRemapTableImage;
        Rpp32f* colRemapTableRow = colRemapTableImage;

#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++) {
            Rpp16f* dstPtrTemp = dstPtrRow + dstLocRow * dstDescPtr->strides.hStride;
            Rpp32f* rowRemapTableTemp =
                rowRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;
            Rpp32f* colRemapTableTemp =
                colRemapTableRow + dstLocRow * remapTableDescPtr->strides.hStride;

            int vectorLoopCount = 0;
#if __AVX2__
            __m256 pBilinearCoeffs[4];
            RpptBilinearNbhoodLocsVecLen8 srcLocs;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel) {
                __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                __m256 pSrc[4], pDst;
                compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs,
                                                        pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp16f>, srcPtrChannel, srcDescPtr,
                              srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);
                compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTemp, pDst);
                dstPtrTemp += vectorIncrementPerChannel;
                rowRemapTableTemp += vectorIncrementPerChannel;
                colRemapTableTemp += vectorIncrementPerChannel;
            }
#endif
            for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                compute_generic_bilinear_interpolation_pln_to_pln(
                    *rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++,
                    srcPtrChannel, srcDescPtr, dstDescPtr);
        }
    }
}
RppStatus remap_bilinear_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                             RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                             Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                             RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp16f* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* rowRemapTableImage =
            rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32f* colRemapTableImage =
            colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        remap_bilinear_f16_f16_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                         rowRemapTableImage, colRemapTableImage, remapTableDescPtr,
                                         roi, roiLTRB, intraThreads);
    }

    return RPP_SUCCESS;
}

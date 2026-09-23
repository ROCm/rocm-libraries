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
#include "rpp_cpu_filter.hpp"

#define MAX_FILTER_SIZE 81  // Maximum kernel size is 9x9 = 81 coefficients

namespace rpp_gaussian_filter {

inline Rpp32f gaussian(int iSquare, int j, Rpp32f mulFactor) {
    Rpp32f expFactor = -(iSquare + (j * j)) * mulFactor;
    expFactor = std::exp(expFactor);
    return expFactor;
}

inline void create_gaussian_kernel_host(Rpp32f* filter, Rpp32f stdDev, int kernelSize) {
    int kernelHalfSize = kernelSize / 2;
    Rpp32f mulFactor = 1.0f / (2.0f * stdDev * stdDev);
    int rowIdx = 0;

    // Compute values for only top left quarter and replicate the values
    // Combine kernel generation with sum computation for efficiency
    Rpp32f kernelSum = 0.0f;
    for (int i = -kernelHalfSize; i <= 0; i++, rowIdx += kernelSize) {
        int iSquare = i * i;
        Rpp32f rowSum = 0.0f;
        for (int j = -kernelHalfSize; j <= 0; j++) {
            Rpp32f val = gaussian(iSquare, j, mulFactor);
            filter[rowIdx + (kernelHalfSize + j)] = filter[rowIdx + (kernelHalfSize - j)] = val;
            rowSum += (j == 0) ? val : 2.0f * val;  // center once, mirrored pair twice
        }
        // If this row was memcpy'd to a mirrored row, count it twice
        bool mirrored = ((kernelSize * (kernelSize - 1) - rowIdx) != rowIdx);
        kernelSum += mirrored ? 2.0f * rowSum : rowSum;
        if (mirrored)
            std::memcpy(&filter[kernelSize * (kernelSize - 1) - rowIdx], &filter[rowIdx],
                        kernelSize * sizeof(float));
    }

    // Normalize the kernel
    Rpp32f invSum = 1.0f / kernelSum;
    for (int i = 0; i < kernelSize * kernelSize; i++) filter[i] *= invSum;
}

}  // namespace rpp_gaussian_filter

using namespace rpp_gaussian_filter;

// -------------------- Per-image AVX fast-path implementation --------------------

template <typename T>
static inline RppStatus gaussian_filter_host_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                  T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                  Rpp32f* filterTensor, Rpp32u kernelSize,
                                                  RpptROI roi, RppLayoutParams layoutParams
#if __AVX2__
                                                  ,
                                                  __m256* pFilter, __m256i* pxMaskPln,
                                                  __m256i* pxMaskPkd
#endif
) {
    Rpp32u padLength = kernelSize / 2;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
    Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    if (kernelSize == 3) {
        T *srcPtrRow[3], *dstPtrRow;
        for (int i = 0; i < 3; i++) srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;

        // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude 2 * padLength number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 14) * 14;
            for (int c = 0; c < srcDescPtr->c; c++) {
                srcPtrRow[0] = srcPtrChannel;
                srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength);
                    T* srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T* dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    RpptImageBorderEdge padVertical = i < padLength
                                                          ? RpptImageBorderEdge::TOP_EDGE
                                                          : RpptImageBorderEdge::BOTTOM_EDGE;
                    process_left_border_columns_pln_pln(
                        srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                        rowKernelLoopLimit, filterTensor, padVertical);
                    dstPtrTemp += padLength;
#if __AVX2__
                    Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE)
                                          ? rowKernelLoopLimit - 1
                                          : 0;
                    // Prefetch next row for better cache performance
                    if (i + 1 < roi.xywhROI.roiHeight) {
                        int prefetchCount = (kernelSize < 3) ? kernelSize : 3;
                        for (int k = 0; k < prefetchCount; k++)
                            _mm_prefetch((const char*)(srcPtrRow[k] + srcDescPtr->strides.hStride),
                                         _MM_HINT_T0);
                    }
                    // process alignedLength number of columns in each row - alignedLength set based
                    // on convolution operations per pass
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 14) {
                        __m256 pRow[6], pDst[2];
                        rpp_load_filter_NxN_pln_host<3>(pRow, srcPtrTemp, rowKernelLoopLimit,
                                                        padIndex);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;

                        // Unrolled convolution loop for 3x3 kernel
                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[0], pRow[0], pRow[1], &pFilter[0],
                                                          pxMaskPln);
                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[1], pRow[1], avx_p0, &pFilter[0],
                                                          pxMaskPln);

                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[0], pRow[2], pRow[3], &pFilter[3],
                                                          pxMaskPln);
                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[1], pRow[3], avx_p0, &pFilter[3],
                                                          pxMaskPln);

                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[0], pRow[4], pRow[5], &pFilter[6],
                                                          pxMaskPln);
                        permute_blend_add_3x3<1, 3, 0, 1>(pDst[1], pRow[5], avx_p0, &pFilter[6],
                                                          pxMaskPln);

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                        // In each pass, convolution filter is applied 14 times
                        increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                        dstPtrTemp += 14;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to
                    // next rows
                    increment_row_ptrs(srcPtrRow, kernelSize,
                                       (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize,
                                                    padLength, unpaddedWidth, rowKernelLoopLimit,
                                                    filterTensor, padVertical);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256 pRow[12], pDst[3];
                    rpp_load_filter_NxN_pkd_host<3>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);

                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    pDst[2] = avx_p0;

                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3;
                         k++, filterIndex += 3, rowIndex += 4) {
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[0], pRow[rowIndex],
                                                           pRow[rowIndex + 1],
                                                           &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[1], pRow[rowIndex + 1],
                                                           pRow[rowIndex + 2],
                                                           &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[2], pRow[rowIndex + 2],
                                                           pRow[rowIndex + 3],
                                                           &pFilter[filterIndex], pxMaskPkd);
                    }

                    // In each pass, convolution filter is applied 24 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_store24_f32_to_f32_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_store24_f32_to_f16_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_store24_f32_to_i8_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_store24_f32_to_u8_avx(dstPtrTemp, pDst);
                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1],
                                            dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, filterTensor, padVertical);
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256 pRow[12], pDst[3];
                    rpp_load_filter_NxN_pkd_host<3>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);

                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    pDst[2] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3;
                         k++, filterIndex += 3, rowIndex += 4) {
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[0], pRow[rowIndex],
                                                           pRow[rowIndex + 1],
                                                           &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[1], pRow[rowIndex + 1],
                                                           pRow[rowIndex + 2],
                                                           &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[2], pRow[rowIndex + 2],
                                                           pRow[rowIndex + 3],
                                                           &pFilter[filterIndex], pxMaskPkd);
                    }

                    __m128 pDstPln[6];
                    rpp_convert24_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store24_float_pkd_pln(dstPtrTempChannels, pDstPln);
                    // In each pass, convolution filter is applied 24 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                    increment_row_ptrs(dstPtrTempChannels, kernelSize, 8);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++) {
                    int channel = c % 3;
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize,
                        padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
                since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 14) * 14;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3][3] = {{srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                       {srcPtrRow[0] + srcDescPtr->strides.cStride,
                                        srcPtrRow[1] + srcDescPtr->strides.cStride,
                                        srcPtrRow[2] + srcDescPtr->strides.cStride},
                                       {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRow[1] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}};

                T* dstPtrTemp = dstPtrRow;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 3x3 box,
                // process them separately
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth,
                            rowKernelLoopLimit, filterTensor, 1, padVertical,
                            RpptImageBorderEdge::LEFT_EDGE);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process alignedLength number of columns in each row - alignedLength set based on
                // convolution operations per pass
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 14) {
                    __m256 pResult[6];
                    for (int c = 0; c < 3; c++) {
                        int channelStride = c * 2;
                        __m256 pRow[6];
                        rpp_load_filter_NxN_pln_host<3>(pRow, srcPtrTemp[c], rowKernelLoopLimit,
                                                        padIndex);
                        pResult[channelStride] = avx_p0;
                        pResult[channelStride + 1] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3;
                             k++, filterIndex += 3, rowIndex += 2) {
                            permute_blend_add_3x3<1, 3, 0, 1>(pResult[channelStride],
                                                              pRow[rowIndex], pRow[rowIndex + 1],
                                                              &pFilter[filterIndex], pxMaskPln);
                            permute_blend_add_3x3<1, 3, 0, 1>(pResult[channelStride + 1],
                                                              pRow[rowIndex + 1], avx_p0,
                                                              &pFilter[filterIndex], pxMaskPln);
                        }

                        // In each pass, convolution filter is applied 14 times
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 14);
                    }
                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResult);
                    dstPtrTemp += 42;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < 3; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    } else if (kernelSize == 5) {
        T *srcPtrRow[5], *dstPtrRow;
        for (int i = 0; i < 5; i++) srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;

        // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
                since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                srcPtrRow[0] = srcPtrChannel;
                for (int k = 1; k < 5; k++)
                    srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;

                dstPtrRow = dstPtrChannel;
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1 : 0;
                    T* srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3],
                                        srcPtrRow[4]};
                    T* dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    RpptImageBorderEdge padVertical = i < padLength
                                                          ? RpptImageBorderEdge::TOP_EDGE
                                                          : RpptImageBorderEdge::BOTTOM_EDGE;
                    process_left_border_columns_pln_pln(
                        srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                        rowKernelLoopLimit, filterTensor, padVertical);
                    dstPtrTemp += padLength;
#if __AVX2__
                    Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE)
                                          ? rowKernelLoopLimit - 1
                                          : 0;
                    // process alignedLength number of columns in each row - alignedLength set based
                    // on convolution operations per pass
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                        __m256 pRow[10], pDst[2];
                        rpp_load_filter_NxN_pln_host<5>(pRow, srcPtrTemp, rowKernelLoopLimit,
                                                        padIndex);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5;
                             k++, filterIndex += 5, rowIndex += 2) {
                            permute_blend_add_5x5_pln(pDst[0], pRow[rowIndex], pRow[rowIndex + 1],
                                                      &pFilter[filterIndex]);
                            permute_blend_add_5x5_pln(pDst[1], pRow[rowIndex + 1], avx_p0,
                                                      &pFilter[filterIndex]);
                        }

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                        // In each pass, convolution filter is applied 12 times
                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        dstPtrTemp += 12;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to
                    // next rows
                    increment_row_ptrs(srcPtrRow, kernelSize,
                                       (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength * 3)) / 32) * 32;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize,
                                                    padLength, unpaddedWidth, rowKernelLoopLimit,
                                                    filterTensor, padVertical);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                    __m256 pRow[20], pDst[2];
                    rpp_load_filter_NxN_pkd_host<5>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5;
                         k++, filterIndex += 5, rowIndex += 4) {
                        permute_blend_add_5x5_pkd(pDst[0], &pRow[rowIndex], &pFilter[filterIndex]);
                        permute_blend_add_5x5_pkd(pDst[1], &pRow[rowIndex + 1],
                                                  &pFilter[filterIndex]);
                    }

                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                    // In each pass, convolution filter is applied 16 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                    dstPtrTemp += 16;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
                since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3][5];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 5; k++) srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrRow;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth,
                            rowKernelLoopLimit, filterTensor, 1, padVertical,
                            RpptImageBorderEdge::LEFT_EDGE);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process alignedLength number of columns in each row - alignedLength set based on
                // convolution operations per pass
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256 pRow[10];
                        rpp_load_filter_NxN_pln_host<5>(pRow, srcPtrTemp[c], rowKernelLoopLimit,
                                                        padIndex);
                        pResultPln[c] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5;
                             k++, filterIndex += 5, rowIndex += 2)
                            permute_blend_add_5x5_pln(pResultPln[c], pRow[rowIndex],
                                                      pRow[rowIndex + 1], &pFilter[filterIndex]);
                        // In each pass, convolution filter is applied 8 times
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                    }

                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1],
                                            dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, filterTensor, padVertical);
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[20], pDst[2];
                    rpp_load_filter_NxN_pkd_host<5>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5;
                         k++, filterIndex += 5, rowIndex += 4) {
                        permute_blend_add_5x5_pkd(pDst[0], &pRow[rowIndex], &pFilter[filterIndex]);
                        permute_blend_add_5x5_pkd(pDst[1], &pRow[rowIndex + 1],
                                                  &pFilter[filterIndex]);
                    }

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    // In each pass, convolution filter is applied 12 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize,
                        padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
    } else if (kernelSize == 7) {
        T *srcPtrRow[7], *dstPtrRow;
        for (int i = 0; i < 7; i++) srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;

        // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                srcPtrRow[0] = srcPtrChannel;
                for (int k = 1; k < 7; k++)
                    srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;

                dstPtrRow = dstPtrChannel;
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1 : 0;
                    T* srcPtrTemp[7];
                    for (int k = 0; k < 7; k++) srcPtrTemp[k] = srcPtrRow[k];
                    T* dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    RpptImageBorderEdge padVertical = i < padLength
                                                          ? RpptImageBorderEdge::TOP_EDGE
                                                          : RpptImageBorderEdge::BOTTOM_EDGE;
                    process_left_border_columns_pln_pln(
                        srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                        rowKernelLoopLimit, filterTensor, padVertical);
                    dstPtrTemp += padLength;
#if __AVX2__
                    Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE)
                                          ? rowKernelLoopLimit - 1
                                          : 0;
                    // process alignedLength number of columns in each row - alignedLength set based
                    // on convolution operations per pass
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        __m256 pRow[14], pDst;
                        rpp_load_filter_NxN_pln_host<7>(pRow, srcPtrTemp, rowKernelLoopLimit,
                                                        padIndex);
                        pDst = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7;
                             k++, filterIndex += 7, rowIndex += 2)
                            permute_blend_add_7x7_pln(pDst, &pRow[rowIndex], &pFilter[filterIndex]);

                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128(
                                (__m128i*)dstPtrTemp,
                                _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                        // In each pass, convolution filter is applied 8 times
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to
                    // next rows
                    increment_row_ptrs(srcPtrRow, kernelSize,
                                       (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize,
                                                    padLength, unpaddedWidth, rowKernelLoopLimit,
                                                    filterTensor, padVertical);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pRow[28], pDst;
                    rpp_load_filter_NxN_pkd_host<7>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                    pDst = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7;
                         k++, filterIndex += 7, rowIndex += 4)
                        permute_blend_add_7x7_pkd(pDst, &pRow[rowIndex], pRow[rowIndex + 3],
                                                  &pFilter[filterIndex]);

                    if constexpr (std::is_same<T, Rpp32f>::value)
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        _mm_storeu_si128(
                            (__m128i*)dstPtrTemp,
                            _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                    // In each pass, convolution filter is applied 8 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    dstPtrTemp += 8;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3][7];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 7; k++) srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrRow;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth,
                            rowKernelLoopLimit, filterTensor, 1, padVertical,
                            RpptImageBorderEdge::LEFT_EDGE);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process alignedLength number of columns in each row - alignedLength set based on
                // convolution operations per pass
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256 pRow[14];
                        rpp_load_filter_NxN_pln_host<7>(pRow, srcPtrTemp[c], rowKernelLoopLimit,
                                                        padIndex);
                        pResultPln[c] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7;
                             k++, filterIndex += 7, rowIndex += 2)
                            permute_blend_add_7x7_pln(pResultPln[c], &pRow[rowIndex],
                                                      &pFilter[filterIndex]);
                        // In each pass, convolution filter is applied 8 times
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                    }
                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1],
                                            dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, filterTensor, padVertical);
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[28], pDst[2];
                    rpp_load_filter_NxN_pkd_host<7>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7;
                         k++, filterIndex += 7, rowIndex += 4) {
                        permute_blend_add_7x7_pkd(pDst[0], &pRow[rowIndex], pRow[rowIndex + 3],
                                                  &pFilter[filterIndex]);
                        permute_blend_add_7x7_pkd(pDst[1], &pRow[rowIndex + 1], avx_p0,
                                                  &pFilter[filterIndex]);
                    }

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    // In each pass, convolution filter is applied 12 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize,
                        padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
    } else if (kernelSize == 9) {
        T *srcPtrRow[9], *dstPtrRow;
        for (int i = 0; i < 9; i++) srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;

        // box filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                srcPtrRow[0] = srcPtrChannel;
                for (int k = 1; k < 9; k++)
                    srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1 : 0;
                    T* srcPtrTemp[9];
                    for (int k = 0; k < 9; k++) srcPtrTemp[k] = srcPtrRow[k];
                    T* dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    RpptImageBorderEdge padVertical = i < padLength
                                                          ? RpptImageBorderEdge::TOP_EDGE
                                                          : RpptImageBorderEdge::BOTTOM_EDGE;
                    process_left_border_columns_pln_pln(
                        srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth,
                        rowKernelLoopLimit, filterTensor, padVertical);
                    dstPtrTemp += padLength;
#if __AVX2__
                    Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE)
                                          ? rowKernelLoopLimit - 1
                                          : 0;
                    // process alignedLength number of columns in each row - alignedLength set based
                    // on convolution operations per pass
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        __m256 pRow[18], pDst;
                        rpp_load_filter_NxN_pln_host<9>(pRow, srcPtrTemp, rowKernelLoopLimit,
                                                        padIndex);
                        pDst = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 9;
                             k++, filterIndex += 9, rowIndex += 2)
                            permute_blend_add_9x9_pln(pDst, &pRow[rowIndex], &pFilter[filterIndex]);

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128(
                                (__m128i*)dstPtrTemp,
                                _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                        // In each pass, convolution filter is applied 8 times
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to
                    // next rows
                    increment_row_ptrs(srcPtrRow, kernelSize,
                                       (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize,
                                                    padLength, unpaddedWidth, rowKernelLoopLimit,
                                                    filterTensor, padVertical);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pRow[36], pDst;
                    rpp_load_filter_NxN_pkd_host<9>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                    pDst = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 9;
                         k++, filterIndex += 9, rowIndex += 4)
                        permute_blend_add_9x9_pkd(pDst, &pRow[rowIndex], &pFilter[filterIndex]);

                    if constexpr (std::is_same<T, Rpp32f>::value)
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        _mm_storeu_si128(
                            (__m128i*)dstPtrTemp,
                            _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                    // In each pass, convolution filter is applied 8 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    dstPtrTemp += 8;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // gaussian filter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[3][9];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 9; k++) srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrRow;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth,
                            rowKernelLoopLimit, filterTensor, 1, padVertical,
                            RpptImageBorderEdge::LEFT_EDGE);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                // process alignedLength number of columns in each row - alignedLength set based on
                // convolution operations per pass
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256 pRow[18];
                        rpp_load_filter_NxN_pln_host<9>(pRow, srcPtrTemp[c], rowKernelLoopLimit,
                                                        padIndex);
                        pResultPln[c] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 9;
                             k++, filterIndex += 9, rowIndex += 2)
                            permute_blend_add_9x9_pln(pResultPln[c], &pRow[rowIndex],
                                                      &pFilter[filterIndex]);
                        // In each pass, convolution filter is applied 8 times
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                    }

                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        convolution_filter_generic_tensor(
                            srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                            unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 40) * 40;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1 : 0;
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1],
                                            dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit, filterTensor, padVertical);
#if __AVX2__
                Rpp32s padIndex =
                    (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ? rowKernelLoopLimit - 1 : 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[45], pDst[2];
                    rpp_load_gaussian_filter_9x9_pkd_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit,
                                                              padIndex);
                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 9;
                         k++, filterIndex += 9, rowIndex += 5) {
                        permute_blend_add_9x9_pkd(pDst[0], &pRow[rowIndex], &pFilter[filterIndex]);
                        permute_blend_add_9x9_pkd(pDst[1], &pRow[rowIndex + 1],
                                                  &pFilter[filterIndex]);
                    }

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    // In each pass, convolution filter is applied 12 times
                    increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize,
                        padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next
                // rows
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus gaussian_filter_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                      Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType, RppLayoutParams layoutParams,
                                      rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return gaussian_filter_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                   stdDevTensor, kernelSize, roiTensorPtrSrc,
                                                   roiType, layoutParams, handle);

#if __AVX2__
    // Set shuffle masks shared across all kernel sizes
    __m256i pxMaskPln[7] = {avx_pxMaskRotate0To1, avx_pxMaskRotate0To2, avx_pxMaskRotate0To3,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To5, avx_pxMaskRotate0To6,
                            avx_pxMaskRotate0To7};
    __m256i pxMaskPkd[7] = {avx_pxMaskRotate0To3, avx_pxMaskRotate0To6, avx_pxMaskRotate0To1,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To7, avx_pxMaskRotate0To2,
                            avx_pxMaskRotate0To5};

    // Pre-allocate and broadcast filter coefficients for all batches before the parallel loop
    __m256* pFilterBatch =
        (__m256*)aligned_alloc(32, dstDescPtr->n * MAX_FILTER_SIZE * sizeof(__m256));
    if (!pFilterBatch)
        return gaussian_filter_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                   stdDevTensor, kernelSize, roiTensorPtrSrc,
                                                   roiType, layoutParams, handle);

    int filterSize = kernelSize * kernelSize;
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        Rpp32f* filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost +
                               batchCount * kernelSize * kernelSize;
        create_gaussian_kernel_host(filterTensor, stdDevTensor[batchCount], kernelSize);
        __m256* pFilter = pFilterBatch + batchCount * MAX_FILTER_SIZE;
        for (int i = 0; i < filterSize; i++) pFilter[i] = _mm256_set1_ps(filterTensor[i]);
    }
#endif

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f* filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost +
                               batchCount * kernelSize * kernelSize;

#if __AVX2__
        __m256* pFilter = pFilterBatch + batchCount * MAX_FILTER_SIZE;
        gaussian_filter_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, filterTensor,
                                  kernelSize, roi, layoutParams, pFilter, pxMaskPln, pxMaskPkd);
#else
        gaussian_filter_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, filterTensor,
                                  kernelSize, roi, layoutParams);
#endif
    }

#if __AVX2__
    free(pFilterBatch);
#endif
    return RPP_SUCCESS;
}

// -------------------- Per-image generic (scalar) implementation --------------------

template <typename T>
static inline RppStatus gaussian_filter_generic_host_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                          T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                          Rpp32f* filterTensor, Rpp32u kernelSize,
                                                          RpptROI roi,
                                                          RppLayoutParams layoutParams) {
    Rpp32u padLength = kernelSize / 2;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
    Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    T *srcPtrRow[kernelSize], *dstPtrRow;
    for (int k = 0; k < kernelSize; k++)
        srcPtrRow[k] = srcPtrChannel + k * srcDescPtr->strides.hStride;
    dstPtrRow = dstPtrChannel;

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        for (int c = 0; c < srcDescPtr->c; c++) {
            srcPtrRow[0] = srcPtrChannel;
            for (int k = 1; k < kernelSize; k++)
                srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength);
                T* srcPtrTemp[kernelSize];
                for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
                T* dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE
                                                                : RpptImageBorderEdge::BOTTOM_EDGE;
                process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength,
                                                    unpaddedWidth, rowKernelLoopLimit, filterTensor,
                                                    padVertical);
                dstPtrTemp += padLength;
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                increment_row_ptrs(srcPtrRow, kernelSize,
                                   (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
            T* dstPtrTemp = dstPtrRow;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize,
                                                padLength, unpaddedWidth, rowKernelLoopLimit,
                                                filterTensor, padVertical);
            dstPtrTemp += padLength * 3;
            vectorLoopCount += padLength * 3;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3,
                                                  kernelSize, padLength, unpaddedWidth,
                                                  rowKernelLoopLimit, filterTensor, 3, padVertical);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTemp++;
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[3][kernelSize];
            for (int c = 0; c < 3; c++) {
                Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
            }
            T* dstPtrTemp = dstPtrRow;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            for (int k = 0; k < padLength; k++) {
                for (int c = 0; c < 3; c++) {
                    convolution_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize,
                                                      padLength, unpaddedWidth, rowKernelLoopLimit,
                                                      filterTensor, 1, padVertical,
                                                      RpptImageBorderEdge::LEFT_EDGE);
                    dstPtrTemp++;
                }
            }
            vectorLoopCount += padLength;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                for (int c = 0; c < srcDescPtr->c; c++) {
                    convolution_filter_generic_tensor(
                        srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength,
                        unpaddedWidth, rowKernelLoopLimit, filterTensor, 1, padVertical);
                    increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                    dstPtrTemp++;
                }
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        T* dstPtrChannels[3];
        for (int c = 0; c < 3; c++)
            dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            int vectorLoopCount = 0;
            bool padLengthRows = (i < padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
            T* dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            RpptImageBorderEdge padVertical =
                i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
            process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels,
                                                kernelSize, padLength, unpaddedWidth,
                                                rowKernelLoopLimit, filterTensor, padVertical);
            vectorLoopCount += padLength * 3;
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                int channel = vectorLoopCount % 3;
                convolution_filter_generic_tensor(
                    srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize,
                    padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3, padVertical);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTempChannels[channel]++;
            }
            increment_row_ptrs(srcPtrRow, kernelSize,
                               (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
            increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
        }
    }
    return RPP_SUCCESS;
}

template <typename T>
RppStatus gaussian_filter_generic_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                              RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                              Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
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

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f* filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost +
                               batchCount * kernelSize * kernelSize;
        create_gaussian_kernel_host(filterTensor, stdDevTensor[batchCount], kernelSize);
        gaussian_filter_generic_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                          filterTensor, kernelSize, roi, layoutParams);
    }
    return RPP_SUCCESS;
}

// ==================== SINGLE IMAGE PROCESSING ====================

template <typename T>
RppStatus gaussian_filter_host_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                            RpptDescPtr dstDescPtr, Rpp32f stdDev,
                                            Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType, RppLayoutParams layoutParams,
                                            rpp::Handle& handle) {
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return gaussian_filter_generic_host_single_image(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                         stdDev, kernelSize, roiTensorPtrSrc,
                                                         roiType, layoutParams, handle);

    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(roiTensorPtrSrc, &roi, &roiDefault, roiType);

    Rpp32f filterTensor[MAX_FILTER_SIZE];
    create_gaussian_kernel_host(filterTensor, stdDev, kernelSize);

#if __AVX2__
    __m256i pxMaskPln[7] = {avx_pxMaskRotate0To1, avx_pxMaskRotate0To2, avx_pxMaskRotate0To3,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To5, avx_pxMaskRotate0To6,
                            avx_pxMaskRotate0To7};
    __m256i pxMaskPkd[7] = {avx_pxMaskRotate0To3, avx_pxMaskRotate0To6, avx_pxMaskRotate0To1,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To7, avx_pxMaskRotate0To2,
                            avx_pxMaskRotate0To5};

    int filterSize = kernelSize * kernelSize;
    __m256 pFilterArr[MAX_FILTER_SIZE];
    for (int i = 0; i < filterSize; i++) pFilterArr[i] = _mm256_set1_ps(filterTensor[i]);

    gaussian_filter_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, filterTensor, kernelSize, roi,
                              layoutParams, pFilterArr, pxMaskPln, pxMaskPkd);
    return RPP_SUCCESS;
#else
    return gaussian_filter_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, filterTensor,
                                     kernelSize, roi, layoutParams);
#endif
}

template <typename T>
RppStatus gaussian_filter_generic_host_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                    RpptDescPtr dstDescPtr, Rpp32f stdDev,
                                                    Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(roiTensorPtrSrc, &roi, &roiDefault, roiType);
    Rpp32f* filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
    create_gaussian_kernel_host(filterTensor, stdDev, kernelSize);
    return gaussian_filter_generic_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, filterTensor,
                                             kernelSize, roi, layoutParams);
}

template RppStatus gaussian_filter_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                      Rpp32f*, Rpp32u, RpptROIPtr, RpptRoiType,
                                                      RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                       Rpp32f*, Rpp32u, RpptROIPtr, RpptRoiType,
                                                       RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*, RpptDescPtr,
                                                       Rpp32f*, Rpp32u, RpptROIPtr, RpptRoiType,
                                                       RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                      Rpp32f*, Rpp32u, RpptROIPtr, RpptRoiType,
                                                      RppLayoutParams, rpp::Handle&);

template RppStatus gaussian_filter_host_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                            RpptDescPtr, Rpp32f, Rpp32u, RpptROIPtr,
                                                            RpptRoiType, RppLayoutParams,
                                                            rpp::Handle&);
template RppStatus gaussian_filter_host_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                             RpptDescPtr, Rpp32f, Rpp32u,
                                                             RpptROIPtr, RpptRoiType,
                                                             RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_host_single_image<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*,
                                                             RpptDescPtr, Rpp32f, Rpp32u,
                                                             RpptROIPtr, RpptRoiType,
                                                             RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_host_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                            RpptDescPtr, Rpp32f, Rpp32u, RpptROIPtr,
                                                            RpptRoiType, RppLayoutParams,
                                                            rpp::Handle&);
template RppStatus gaussian_filter_generic_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                              RpptDescPtr, Rpp32f*, Rpp32u,
                                                              RpptROIPtr, RpptRoiType,
                                                              RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                               RpptDescPtr, Rpp32f*, Rpp32u,
                                                               RpptROIPtr, RpptRoiType,
                                                               RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*,
                                                               RpptDescPtr, Rpp32f*, Rpp32u,
                                                               RpptROIPtr, RpptRoiType,
                                                               RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                              RpptDescPtr, Rpp32f*, Rpp32u,
                                                              RpptROIPtr, RpptRoiType,
                                                              RppLayoutParams, rpp::Handle&);

template RppStatus gaussian_filter_generic_host_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                                    RpptDescPtr, Rpp32f, Rpp32u,
                                                                    RpptROIPtr, RpptRoiType,
                                                                    RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                                     RpptDescPtr, Rpp32f, Rpp32u,
                                                                     RpptROIPtr, RpptRoiType,
                                                                     RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_single_image<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*,
                                                                     RpptDescPtr, Rpp32f, Rpp32u,
                                                                     RpptROIPtr, RpptRoiType,
                                                                     RppLayoutParams, rpp::Handle&);
template RppStatus gaussian_filter_generic_host_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                                    RpptDescPtr, Rpp32f, Rpp32u,
                                                                    RpptROIPtr, RpptRoiType,
                                                                    RppLayoutParams, rpp::Handle&);

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

namespace rpp_erode {

// generic raw c code for erode
template <typename T>
inline void erode_generic_tensor(T** srcPtrTemp, T* dstPtrTemp, Rpp32s columnIndex,
                                 Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth,
                                 Rpp32s rowKernelLoopLimit, Rpp32u channels = 1) {
    // Initialize result to maximum value
    T result;
    if constexpr (std::is_same<T, Rpp8u>::value)
        result = static_cast<T>(255);
    else if constexpr (std::is_same<T, Rpp8s>::value)
        result = static_cast<T>(127);
    else if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
        result = static_cast<T>(1.0);
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    for (int i = 0; i < rowKernelLoopLimit; i++)
        for (int j = 0, k = 0; j < columnKernelLoopLimit; j++, k += channels)
            result = std::min<T>(result, srcPtrTemp[i][k]);
    *dstPtrTemp = result;
}

// process padLength number of columns in each row for PLN-PLN case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process
// them separately
template <typename T>
inline void process_left_border_columns_pln_pln(T** srcPtrTemp, T* dstPtrTemp, Rpp32u kernelSize,
                                                Rpp32u padLength, Rpp32u unpaddedWidth,
                                                Rpp32s rowKernelLoopLimit) {
    for (int k = 0; k < padLength; k++) {
        erode_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth,
                             rowKernelLoopLimit);
        dstPtrTemp++;
    }
}

// process padLength * 3 number of columns in each row for PKD-PKD case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process
// them separately
template <typename T>
inline void process_left_border_columns_pkd_pkd(T** srcPtrTemp, T** srcPtrRow, T* dstPtrTemp,
                                                Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit) {
    for (int c = 0; c < 3; c++) {
        T* dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++) {
            erode_generic_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength,
                                 unpaddedWidth, rowKernelLoopLimit, 3);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

// process padLength * 3 number of columns in each row for PKD-PLN case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process
// them separately
template <typename T>
inline void process_left_border_columns_pkd_pln(T** srcPtrTemp, T** srcPtrRow,
                                                T** dstPtrTempChannels, Rpp32u kernelSize,
                                                Rpp32u padLength, Rpp32u unpaddedWidth,
                                                Rpp32s rowKernelLoopLimit) {
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < padLength; k++) {
            erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength,
                                 unpaddedWidth, rowKernelLoopLimit, 3);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

// -------------------- Set 0 erode compute functions --------------------

// unpack halves from K 256 bit registers and min (used for KxK kernel size U8/I8 variants)
template <const int K>
inline void unpack_and_min_host(__m256i* pxRow, __m256i* pxDst) {
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
#pragma unroll
    for (int i = 1; i < K; i++)
        pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[i], avx_px0));
    pxDst[1] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
#pragma unroll
    for (int i = 1; i < K; i++)
        pxDst[1] = _mm256_min_epi16(pxDst[1], _mm256_unpackhi_epi8(pxRow[i], avx_px0));
}

// unpack and sign extend halves of K 256 bit registers and min (used for KxK kernel size U8/I8
// variants)
template <const int K>
inline void unpack_signext_and_min_host(__m256i* pxRow, __m256i* pxDst) {
    pxDst[0] = _mm256_srai_epi16(_mm256_slli_epi16(_mm256_unpacklo_epi8(pxRow[0], avx_px0), 8), 8);
#pragma unroll
    for (int i = 1; i < K; i++)
        pxDst[0] = _mm256_min_epi16(
            pxDst[0],
            _mm256_srai_epi16(_mm256_slli_epi16(_mm256_unpacklo_epi8(pxRow[i], avx_px0), 8), 8));
    pxDst[1] = _mm256_srai_epi16(_mm256_slli_epi16(_mm256_unpackhi_epi8(pxRow[0], avx_px0), 8), 8);
#pragma unroll
    for (int i = 1; i < K; i++)
        pxDst[1] = _mm256_min_epi16(
            pxDst[1],
            _mm256_srai_epi16(_mm256_slli_epi16(_mm256_unpackhi_epi8(pxRow[i], avx_px0), 8), 8));
}

// min 3 256 bit registers (used for 3x3 kernel size F32/F16 variants)
inline void min_rows_3x3(__m256* pRow, __m256* pDst) {
    pDst[0] = _mm256_min_ps(pRow[0], pRow[1]);
    pDst[0] = _mm256_min_ps(pDst[0], pRow[2]);
}

// min 5 256 bit registers (used for 5x5 kernel size F32/F16 variants)
inline void min_rows_5x5(__m256* pRow, __m256* pDst) {
    pDst[0] = _mm256_min_ps(_mm256_min_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_min_ps(pDst[0], _mm256_min_ps(pRow[3], pRow[4]));
}

// min 7 256 bit registers (used for 7x7 kernel size F32/F16 variants)
inline void min_rows_7x7(__m256* pRow, __m256* pDst) {
    pDst[0] = _mm256_min_ps(_mm256_min_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_min_ps(pDst[0], _mm256_min_ps(_mm256_min_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_min_ps(pDst[0], pRow[6]);
}

// min 9 256 bit registers (used for 9x9 kernel size F32/F16 variants)
inline void min_rows_9x9(__m256* pRow, __m256* pDst) {
    pDst[0] = _mm256_min_ps(_mm256_min_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_min_ps(pDst[0], _mm256_min_ps(_mm256_min_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_min_ps(pDst[0], _mm256_min_ps(_mm256_min_ps(pRow[6], pRow[7]), pRow[8]));
}

}  // namespace rpp_erode

using namespace rpp_erode;

// -------------------- Per-image implementation for erode (char types) --------------------

template <typename T>
static inline RppStatus erode_char_host_tensor_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                    T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                    Rpp32u kernelSize, RpptROI roi,
                                                    RppLayoutParams layoutParams,
                                                    Rpp32u intraThreads
#if __AVX2__
                                                    ,
                                                    __m128i* pxMaskPln, __m128i* pxMaskPkd
#endif
) {
    Rpp32u padLength = kernelSize / 2;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
    Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

#if __AVX2__
    // set the register order needed for blend operations
    Rpp32u blendRegisterOrder[7] = {0, 0, 1, 1, 1, 2, 2};
    if (srcDescPtr->layout == RpptLayout::NCHW) std::fill_n(blendRegisterOrder, 7, 0);
#endif
    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;
    if (kernelSize == 3) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude 2 * padLength number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            /* Note: AVX loop stores 32 bytes but advances by 24, creating 8-byte overrun.
               Subtract 8 to prevent buffer overrun at row boundaries in multithreading */
            Rpp32u reservedLength = (2 * padLength) + 8;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[3];
                    srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                    srcPtrTemp[1] = srcPtrChannel + std::min(windowRow + 1, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                    srcPtrTemp[2] = srcPtrChannel + std::min(windowRow + 2, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                        __m256i pxRow[3], pxRowHalf[2], pxResult;
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 3 loaded row values from
                        // 8 bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<3>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<3>(pxRow, pxRowHalf);

                        // perform blend and shuffle operations to get required order and min of
                        // them
                        __m128i pxTemp[4];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[2], pxMaskPln, blendRegisterOrder);

                        __m128i pxDst[2];
                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }
                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i*)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[3];
                srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrTemp[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrTemp[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrInitial[3] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                int vectorLoopCount = 0;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256i pxRow[3], pxRowHalf[2], pxResult;
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower half and higher half of each of 3 loaded row values from 8
                    // bit to 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<3>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<3>(pxRow, pxRowHalf);

                    // perform blend and shuffle operations for the first 8 output values to get
                    // required order and min of them
                    __m128i pxTemp[4];
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);

                    __m128i pxDst[2];
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                    } else {
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                    }
                    pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                    _mm256_storeu_si256((__m256i*)dstPtrTemp, pxResult);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[3];
                srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrTemp[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrTemp[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrInitial[3] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};
                int vectorLoopCount = 0;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256i pxRow[3], pxRowHalf[2];
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower half and higher half of each of 3 loaded row values from 8
                    // bit to 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<3>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<3>(pxRow, pxRowHalf);

                    // perform blend and shuffle operations for the first 8 output values to get
                    // required order and min of them
                    __m128i pxTemp[4];
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                    blend_shuffle_min_3x3_host<7, 63>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);

                    __m128i pxDst[2];
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                    } else {
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                        pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                    }

                    // convert from PKD3 to PLN3 and store channelwise
                    __m128i pxDstChn[3];
                    rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[0]), pxDstChn[0]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[1]), pxDstChn[1]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[2]), pxDstChn[2]);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                    increment_row_ptrs(dstPtrTempChannels, dstDescPtr->c, 8);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++) {
                    int channel = c % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[3];
                srcPtrRows[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrRows[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrRows[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][3] = {{srcPtrRows[0], srcPtrRows[1], srcPtrRows[2]},
                                       {srcPtrRows[0] + srcDescPtr->strides.cStride,
                                        srcPtrRows[1] + srcDescPtr->strides.cStride,
                                        srcPtrRows[2] + srcDescPtr->strides.cStride},
                                       {srcPtrRows[0] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRows[1] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRows[2] + 2 * srcDescPtr->strides.cStride}};

                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 3x3 box,
                // process them separately
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256i pxResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256i pxRow[3], pxRowHalf[2];
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pxRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 3 loaded row values from
                        // 8 bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<3>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<3>(pxRow, pxRowHalf);

                        // perform blend and shuffle operations for the first 8 output values to
                        // get required order and min of them
                        __m128i pxTemp[4];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<1, 3>(&pxTemp[2], pxMaskPln, blendRegisterOrder);

                        __m128i pxDst[2];
                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }

                        pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                        pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                        pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                    }

                    __m128i pxResultPkd[6];
                    // convert result from pln to pkd format and store in output buffer
                    rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                    _mm_storeu_si128((__m128i*)dstPtrTemp, pxResultPkd[0]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 12), pxResultPkd[1]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 24), pxResultPkd[2]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 36), pxResultPkd[3]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 48), pxResultPkd[4]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 60), pxResultPkd[5]);
                    dstPtrTemp += 72;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        }
    } else if (kernelSize == 5) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            /* Note: AVX loop stores 32 bytes but advances by 24, creating 8-byte overrun.
               Subtract 8 to prevent buffer overrun at row boundaries in multithreading */
            Rpp32u reservedLength = (2 * padLength) + 8;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                        __m256i pxRow[5], pxRowHalf[2], pxResult;
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);

                        // unpack lower and higher half of each of 5 loaded row values from 8
                        // bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<5>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<5>(pxRow, pxRowHalf);

                        __m128i pxTemp[4], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[0], pxMaskPln,
                                                                blendRegisterOrder);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[1], pxMaskPln,
                                                                blendRegisterOrder);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[2], pxMaskPln,
                                                                blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }
                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i*)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 18) * 18;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[5] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 18) {
                    __m256i pxRow[5], pxRowHalf[2];
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower and higher half of each of 5 loaded row values from 8 bit to
                    // 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<5>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<5>(pxRow, pxRowHalf);

                    __m128i pxTemp[5], pxDst[2];
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    pxTemp[4] = xmm_px0;
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[0], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[1], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[2], pxMaskPkd,
                                                             blendRegisterOrder);

                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                    } else {
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_sub_epi8(pxDst[0], _mm256_castsi256_si128(avx_pxConvertI8));
                        pxDst[1] = _mm_sub_epi8(pxDst[1], _mm256_castsi256_si128(avx_pxConvertI8));
                    }

                    // Store 18 bytes: pxDst[0] (16 bytes) + first 2 bytes of pxDst[1]
                    _mm_storeu_si128((__m128i*)dstPtrTemp, pxDst[0]);
                    *(Rpp16s*)(dstPtrTemp + 16) = _mm_extract_epi16(pxDst[1], 0);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                    dstPtrTemp += 18;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 18) * 18;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[5] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 18) {
                    __m256i pxRow[5], pxRowHalf[2];
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower and higher half of each of 5 loaded row values from 8 bit to
                    // 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<5>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<5>(pxRow, pxRowHalf);

                    __m128i pxTemp[5], pxDst[2];
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    pxTemp[4] = xmm_px0;
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[0], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[1], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[2], pxMaskPkd,
                                                             blendRegisterOrder);

                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                    } else {
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                        pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                    }

                    // convert from PKD3 to PLN3 and store channelwise
                    __m128i pxDstChn[3];
                    rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[0]), pxDstChn[0]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[1]), pxDstChn[1]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[2]), pxDstChn[2]);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                    increment_row_ptrs(dstPtrTempChannels, dstDescPtr->c, 6);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++) {
                    int channel = c % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength * 3 number of columns from the beginning and end of each row will
               be computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[5];
                for (int k = 0; k < 5; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][5];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 5; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 5x5 box,
                // process them separately
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256i pxResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256i pxRow[5], pxRowHalf[2];
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pxRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);

                        // unpack lower and higher half of each of 5 loaded row values from 8
                        // bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<5>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<5>(pxRow, pxRowHalf);

                        __m128i pxTemp[4], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[0], pxMaskPln,
                                                                blendRegisterOrder);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[1], pxMaskPln,
                                                                blendRegisterOrder);
                        blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[2], pxMaskPln,
                                                                blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }
                        pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                        pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                        pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                    }

                    __m128i pxResultPkd[6];
                    // convert result from pln to pkd format and store in output buffer
                    rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                    _mm_storeu_si128((__m128i*)dstPtrTemp, pxResultPkd[0]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 12), pxResultPkd[1]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 24), pxResultPkd[2]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 36), pxResultPkd[3]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 48), pxResultPkd[4]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 60), pxResultPkd[5]);
                    dstPtrTemp += 72;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        }
    } else if (kernelSize == 7) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            /* Note: AVX loop stores 32 bytes but advances by 24, creating 8-byte overrun.
               Subtract 8 to prevent buffer overrun at row boundaries in multithreading */
            Rpp32u reservedLength = (2 * padLength) + 8;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                        __m256i pxRow[7], pxRowHalf[2], pxResult;
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);

                        // unpack lower and higher half of each of 7 loaded row values from 8
                        // bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<7>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<7>(pxRow, pxRowHalf);

                        __m128i pxTemp[4], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[0], pxMaskPln,
                                                                        blendRegisterOrder);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[1], pxMaskPln,
                                                                        blendRegisterOrder);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[2], pxMaskPln,
                                                                        blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }
                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i*)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[7];
                for (int k = 0; k < 7; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][7];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 7; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 7x7 box,
                // process them separately
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    __m256i pxResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256i pxRow[7], pxRowHalf[2];
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pxRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);

                        // unpack lower and higher half of each of 7 loaded row values from 8
                        // bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<7>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<7>(pxRow, pxRowHalf);

                        __m128i pxTemp[4], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[0], pxMaskPln,
                                                                        blendRegisterOrder);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[1], pxMaskPln,
                                                                        blendRegisterOrder);
                        blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[2], pxMaskPln,
                                                                        blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value) {
                            pxDst[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packs_epi16(pxTemp[2], xmm_px0);
                        } else {
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        }
                        pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                        pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                        pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                    }

                    __m128i pxResultPkd[6];
                    // convert result from pln to pkd format and store in output buffer
                    rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                    _mm_storeu_si128((__m128i*)dstPtrTemp, pxResultPkd[0]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 12), pxResultPkd[1]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 24), pxResultPkd[2]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 36), pxResultPkd[3]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 48), pxResultPkd[4]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 60), pxResultPkd[5]);
                    dstPtrTemp += 72;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = 2 * padLength * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 12) * 12;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[7] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4], srcPtrTemp[5], srcPtrTemp[6]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256i pxRow[7], pxRowHalf[2];
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower and higher half of each of 7 loaded row values from 8 bit to
                    // 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<7>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<7>(pxRow, pxRowHalf);

                    __m128i pxTemp[4], pxResult;
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[0], pxMaskPkd,
                                                                     blendRegisterOrder);
                    blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[1], pxMaskPkd,
                                                                     blendRegisterOrder);
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                    else
                        pxResult = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult = _mm_sub_epi8(pxResult, xmm_pxConvertI8);

                    _mm_storeu_si128((__m128i*)dstPtrTemp, pxResult);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                    dstPtrTemp += 12;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = 2 * padLength * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 12) * 12;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[7] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4], srcPtrTemp[5], srcPtrTemp[6]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256i pxRow[7], pxRowHalf[2];
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower and higher half of each of 7 loaded row values from 8 bit to
                    // 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<7>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<7>(pxRow, pxRowHalf);

                    __m128i pxTemp[4], pxResult[2];
                    extract_4sse_registers(pxRowHalf, pxTemp);
                    blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[0], pxMaskPkd,
                                                                     blendRegisterOrder);
                    blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[1], pxMaskPkd,
                                                                     blendRegisterOrder);

                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult[0] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                    else
                        pxResult[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                    pxResult[1] = xmm_px0;

                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult[0] = _mm_sub_epi8(pxResult[0], xmm_pxConvertI8);

                    // convert from PKD3 to PLN3 and store channelwise
                    __m128i pxDstChn[3];
                    rpp_convert24_pkd3_to_pln3(pxResult[0], pxResult[1], pxDstChn);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[0]), pxDstChn[0]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[1]), pxDstChn[1]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[2]), pxDstChn[2]);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                    increment_row_ptrs(dstPtrTempChannels, dstDescPtr->c, 4);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++) {
                    int channel = c % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        }
    } else if (kernelSize == 9) {
        T *srcPtrRow[9], *dstPtrRow;
        for (int i = 0; i < 9; i++) srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                        __m256i pxRow[9], pxRowHalf[2];
                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 9 loaded row values from
                        // 8 bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<9>(pxRow, pxRowHalf);

                        __m128i pxTemp[3], pxDst;
                        extract_3sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[0], pxMaskPln,
                                                                             blendRegisterOrder);
                        blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[1], pxMaskPln,
                                                                             blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxDst = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        else
                            pxDst = _mm_packus_epi16(pxTemp[0], pxTemp[1]);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxDst = _mm_sub_epi8(pxDst, xmm_pxConvertI8);

                        _mm_storeu_si128((__m128i*)dstPtrTemp, pxDst);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                        dstPtrTemp += 16;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 64) * 64;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[9] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2],
                                       srcPtrTemp[3], srcPtrTemp[4], srcPtrTemp[5],
                                       srcPtrTemp[6], srcPtrTemp[7], srcPtrTemp[8]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // load first 32 elements elements
                __m256i pxRow[9];
                if (alignedLength)
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 32) {
                    __m256i pxRowHalf[2], pxResult;
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<9>(pxRow, pxRowHalf);

                    // get the accumalated result for first 8 elements
                    __m128i px128[8], pxDst[2];
                    extract_4sse_registers(pxRowHalf, &px128[0]);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[0], pxMaskPkd,
                                                                         blendRegisterOrder);

                    // compute for next 8 elements
                    increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<9>(pxRow, pxRowHalf);

                    // get the accumalated result for next 24 elements
                    extract_4sse_registers(pxRowHalf, &px128[4]);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[1], pxMaskPkd,
                                                                         blendRegisterOrder);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[2], pxMaskPkd,
                                                                         blendRegisterOrder);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[3], pxMaskPkd,
                                                                         blendRegisterOrder);

                    // compute final result
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packs_epi16(px128[2], px128[3]);
                    } else {
                        pxDst[0] = _mm_packus_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packus_epi16(px128[2], px128[3]);
                    }
                    pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);

                    if constexpr (std::is_same<T, Rpp8s>::value)
                        pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                    _mm256_storeu_si256((__m256i*)dstPtrTemp, pxResult);
                    dstPtrTemp += 32;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[9];
                for (int k = 0; k < 9; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][9];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 9; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                    __m128i pxResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256i pxRow[9], pxRowHalf[2];
                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 9 loaded row values from
                        // 8 bit to 16 bit and min
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                        else
                            unpack_and_min_host<9>(pxRow, pxRowHalf);

                        __m128i pxTemp[3];
                        extract_3sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[0], pxMaskPln,
                                                                             blendRegisterOrder);
                        blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[1], pxMaskPln,
                                                                             blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResultPln[c] = _mm_packs_epi16(pxTemp[0], pxTemp[1]);
                        else
                            pxResultPln[c] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 16);
                    }
                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxResultPln[0] = _mm_sub_epi8(pxResultPln[0], xmm_pxConvertI8);
                        pxResultPln[1] = _mm_sub_epi8(pxResultPln[1], xmm_pxConvertI8);
                        pxResultPln[2] = _mm_sub_epi8(pxResultPln[2], xmm_pxConvertI8);
                    }

                    __m128i pxResultPkd[4];
                    rpp_convert48_pln3_to_pkd3(pxResultPln, pxResultPkd);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp), pxResultPkd[0]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 12), pxResultPkd[1]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 24), pxResultPkd[2]);
                    _mm_storeu_si128((__m128i*)(dstPtrTemp + 36), pxResultPkd[3]);
                    dstPtrTemp += 48;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 64) * 64;
            T* dstPtrChannels[3];
            for (int c = 0; c < 3; c++)
                dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[9] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2],
                                       srcPtrTemp[3], srcPtrTemp[4], srcPtrTemp[5],
                                       srcPtrTemp[6], srcPtrTemp[7], srcPtrTemp[8]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24) {
                    // load first 32 elements elements
                    __m256i pxRow[9], pxRowHalf[2];
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);

                    // unpack lower half and higher half of each of 9 loaded row values from 8
                    // bit to 16 bit and min
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<9>(pxRow, pxRowHalf);

                    // get the accumalated result for first 8 elements
                    __m128i px128[8], pxDst[2];
                    extract_4sse_registers(pxRowHalf, &px128[0]);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[0], pxMaskPkd,
                                                                         blendRegisterOrder);

                    // compute for next 8 elements
                    increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pxRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    if constexpr (std::is_same<T, Rpp8s>::value)
                        unpack_signext_and_min_host<9>(pxRow, pxRowHalf);
                    else
                        unpack_and_min_host<9>(pxRow, pxRowHalf);

                    // get the accumalated result for next 24 elements
                    extract_4sse_registers(pxRowHalf, &px128[4]);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[1], pxMaskPkd,
                                                                         blendRegisterOrder);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[2], pxMaskPkd,
                                                                         blendRegisterOrder);
                    blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[3], pxMaskPkd,
                                                                         blendRegisterOrder);

                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_packs_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packs_epi16(px128[2], px128[3]);
                    } else {
                        pxDst[0] = _mm_packus_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packus_epi16(px128[2], px128[3]);
                    }

                    if constexpr (std::is_same<T, Rpp8s>::value) {
                        pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                        pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                    }

                    // convert from PKD3 to PLN3 and store
                    __m128i pxDstChn[3];
                    rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[0]), pxDstChn[0]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[1]), pxDstChn[1]);
                    rpp_storeu_si64((__m128i*)(dstPtrTempChannels[2]), pxDstChn[2]);
                    increment_row_ptrs(srcPtrTemp, kernelSize, -8);
                    increment_row_ptrs(dstPtrTempChannels, 3, 8);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

// -------------------- Main erode_char_host_tensor function --------------------

template <typename T>
RppStatus erode_char_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    static_assert((std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value),
                  "T must be Rpp8u or Rpp8s");

    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return erode_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr, kernelSize,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);

        // set the required masks array needed for shuffle operations
#if __AVX2__
    __m128i pxMaskPln[7] = {xmm_pxMaskRotate0To1, xmm_pxMaskRotate0To3, xmm_pxMaskRotate0To5,
                            xmm_pxMaskRotate0To7, xmm_pxMaskRotate0To9, xmm_pxMaskRotate0To11,
                            xmm_pxMaskRotate0To13};
    __m128i pxMaskPkd[7] = {xmm_pxMaskRotate0To5, xmm_pxMaskRotate0To11, xmm_pxMaskRotate0To1,
                            xmm_pxMaskRotate0To7, xmm_pxMaskRotate0To13, xmm_pxMaskRotate0To3,
                            xmm_pxMaskRotate0To9};
#endif

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

#if __AVX2__
        erode_char_host_tensor_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, kernelSize,
                                    roi, layoutParams, intraThreads, pxMaskPln, pxMaskPkd);
#else
        erode_char_host_tensor_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, kernelSize,
                                    roi, layoutParams, intraThreads);
#endif
    }

    return RPP_SUCCESS;
}

// -------------------- Per-image implementation for erode (float types) --------------------

template <typename T>
static inline RppStatus erode_float_host_tensor_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                     T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                     Rpp32u kernelSize, RpptROI roi,
                                                     RppLayoutParams layoutParams,
                                                     Rpp32u intraThreads
#if __AVX2__
                                                     ,
                                                     __m256i* pxMaskPln, __m256i* pxMaskPkd
#endif
) {
    Rpp32u padLength = kernelSize / 2;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
    Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
    Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;
#if __AVX2__
    // set the register order needed for blend operations
    Rpp32u blendRegisterOrder[7] = {0, 0, 1, 1, 1, 2, 2};
    if (srcDescPtr->layout == RpptLayout::NCHW) std::fill_n(blendRegisterOrder, 7, 0);
#endif

    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;
    if (kernelSize == 3) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 12) * 12;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[3];
                    srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                    srcPtrTemp[1] = srcPtrChannel + std::min(windowRow + 1, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                    srcPtrTemp[2] = srcPtrChannel + std::min(windowRow + 2, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                        __m256 pRow[3], pTemp[3], pDst[2];
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_3x3(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_3x3(pRow, &pTemp[1]);
                        pTemp[2] = avx_p0;

                        blend_permute_min_3x3_host<1, 3>(&pTemp[0], &pDst[0], pxMaskPln,
                                                         blendRegisterOrder);
                        blend_permute_min_3x3_host<1, 3>(&pTemp[1], &pDst[1], pxMaskPln,
                                                         blendRegisterOrder);
                        // Store 12 floats: pDst[0] (8 floats) + first 4 floats from pDst[1]
                        if constexpr (std::is_same<T, Rpp32f>::value) {
                            _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                            _mm_storeu_ps(dstPtrTemp + 8, _mm256_castps256_ps128(pDst[1]));
                        } else {  // Rpp16f
                            __m128i pxDst[2];
                            pxDst[0] =
                                _mm256_cvtps_ph(pDst[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                            pxDst[1] =
                                _mm256_cvtps_ph(pDst[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                            _mm_storeu_si128((__m128i*)dstPtrTemp, pxDst[0]);
                            _mm_storel_epi64((__m128i*)(dstPtrTemp + 8), pxDst[1]);
                        }

                        increment_row_ptrs(srcPtrTemp, kernelSize, 4);
                        dstPtrTemp += 12;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[3];
                srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrTemp[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrTemp[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrInitial[3] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                int vectorLoopCount = 0;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16) {
                    __m256 pRow[3], pTemp[3], pDst[2];
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[2]);

                    blend_permute_min_3x3_host<7, 63>(&pTemp[0], &pDst[0], pxMaskPkd,
                                                      blendRegisterOrder);
                    blend_permute_min_3x3_host<7, 63>(&pTemp[1], &pDst[1], pxMaskPkd,
                                                      blendRegisterOrder);

                    rpp_store16_float(dstPtrTemp, pDst);
                    dstPtrTemp += 16;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[3];
                srcPtrTemp[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrTemp[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrTemp[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrInitial[3] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[3], pTemp[3], pDst[2];
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_3x3(pRow, &pTemp[2]);

                    blend_permute_min_3x3_host<7, 63>(&pTemp[0], &pDst[0], pxMaskPkd,
                                                      blendRegisterOrder);
                    blend_permute_min_3x3_host<7, 63>(&pTemp[1], &pDst[1], pxMaskPkd,
                                                      blendRegisterOrder);

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                    increment_row_ptrs(dstPtrTempChannels, dstDescPtr->c, 4);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++) {
                    int channel = c % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            /* Note: AVX loop stores 48 bytes but advances by 42, creating 6-byte overrun.
               Subtract 6 to prevent buffer overrun at row boundaries in multithreading */
            Rpp32u reservedLength = (2 * padLength) + 6;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 14) * 14;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[3];
                srcPtrRows[0] = srcPtrChannel + windowRow * srcDescPtr->strides.hStride;
                srcPtrRows[1] = srcPtrChannel +
                                std::min(windowRow + 1, maxValidRow) * srcDescPtr->strides.hStride;
                srcPtrRows[2] = srcPtrChannel +
                                std::min(windowRow + 2, maxValidRow) * srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][3] = {{srcPtrRows[0], srcPtrRows[1], srcPtrRows[2]},
                                       {srcPtrRows[0] + srcDescPtr->strides.cStride,
                                        srcPtrRows[1] + srcDescPtr->strides.cStride,
                                        srcPtrRows[2] + srcDescPtr->strides.cStride},
                                       {srcPtrRows[0] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRows[1] + 2 * srcDescPtr->strides.cStride,
                                        srcPtrRows[2] + 2 * srcDescPtr->strides.cStride}};

                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 3x3 box,
                // process them separately
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 14) {
                    __m256 pResult[6];
                    for (int c = 0; c < 3; c++) {
                        int channelStride = c * 2;
                        __m256 pRow[3], pTemp[3];
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_3x3(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        rpp_morphological_load_NxN<3, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_3x3(pRow, &pTemp[1]);
                        pTemp[2] = avx_p0;

                        blend_permute_min_3x3_host<1, 3>(&pTemp[0], &pResult[channelStride],
                                                         pxMaskPln, blendRegisterOrder);
                        blend_permute_min_3x3_host<1, 3>(&pTemp[1], &pResult[channelStride + 1],
                                                         pxMaskPln, blendRegisterOrder);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 6);
                    }

                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResult);

                    dstPtrTemp += 42;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        }
    } else if (kernelSize == 5) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            /* Note: AVX loop stores 16 values but advances by 12, creating 4-value overrun.
               Subtract 4 to prevent buffer overrun at row boundaries in multithreading */
            Rpp32u reservedLength = (2 * padLength) + 4;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 12) * 12;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                        __m256 pRow[5], pDst[2], pTemp[3];
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_5x5(pRow, &pTemp[1]);
                        pTemp[2] = avx_p0;

                        blend_permute_min_5x5_host<1, 3, 7, 15>(&pTemp[0], &pDst[0], pxMaskPln,
                                                                blendRegisterOrder);
                        blend_permute_min_5x5_host<1, 3, 7, 15>(&pTemp[1], &pDst[1], pxMaskPln,
                                                                blendRegisterOrder);

                        rpp_store16_float(dstPtrTemp, pDst);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 4);
                        dstPtrTemp += 12;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[5] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    // compute min of loaded values from 9 rows
                    __m256 pRow[5], pDst[2], pTemp[4];
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[2]);
                    pTemp[3] = avx_p0;

                    blend_permute_min_5x5_host<7, 63, 1, 15>(&pTemp[0], &pDst[0], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_permute_min_5x5_host<7, 63, 1, 15>(&pTemp[1], &pDst[1], pxMaskPkd,
                                                             blendRegisterOrder);

                    // Store 12 floats: pDst[0] (8 floats) + first 4 floats from pDst[1]
                    if constexpr (std::is_same<T, Rpp32f>::value) {
                        _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                        _mm_storeu_ps(dstPtrTemp + 8, _mm256_castps256_ps128(pDst[1]));
                    } else {  // Rpp16f
                        __m128i pxDst[2];
                        pxDst[0] = _mm256_cvtps_ph(pDst[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                        pxDst[1] = _mm256_cvtps_ph(pDst[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                        _mm_storeu_si128((__m128i*)dstPtrTemp, pxDst[0]);
                        _mm_storel_epi64((__m128i*)(dstPtrTemp + 8), pxDst[1]);
                    }
                    increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                    dstPtrTemp += 12;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[5];
                for (int k = 0; k < 5; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][5];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 5; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256 pRow[5], pTemp[2];
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_5x5(pRow, &pTemp[1]);
                        blend_permute_min_5x5_host<1, 3, 7, 15>(pTemp, &pResultPln[c], pxMaskPln,
                                                                blendRegisterOrder);
                    }

                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);

                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 24) * 24;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[5];
                for (int k = 0; k < 5; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[5] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    // compute min of loaded values from 9 rows
                    __m256 pRow[5], pDst[2], pTemp[4];
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<5, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_5x5(pRow, &pTemp[2]);
                    pTemp[3] = avx_p0;

                    blend_permute_min_5x5_host<7, 63, 1, 15>(&pTemp[0], &pDst[0], pxMaskPkd,
                                                             blendRegisterOrder);
                    blend_permute_min_5x5_host<7, 63, 1, 15>(&pTemp[1], &pDst[1], pxMaskPkd,
                                                             blendRegisterOrder);

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        }
    } else if (kernelSize == 7) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        __m256 pRow[7], pTemp[2], pDst;
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_7x7(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_7x7(pRow, &pTemp[1]);
                        blend_permute_min_7x7_host<1, 3, 7, 15, 31, 63>(&pTemp[0], &pDst, pxMaskPln,
                                                                        blendRegisterOrder);

                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128(
                                (__m128i*)dstPtrTemp,
                                _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 32) * 32;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[7] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4], srcPtrTemp[5], srcPtrTemp[6]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                __m256 pRow[7], pTemp[4];
                if (alignedLength) {
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[2]);
                }

                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    // compute min of loaded values from 7 rows
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[3]);

                    __m256 pDst;
                    blend_permute_min_7x7_host<7, 63, 1, 15, 127, 3>(pTemp, &pDst, pxMaskPkd,
                                                                     blendRegisterOrder);

                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        _mm_storeu_si128(
                            (__m128i*)dstPtrTemp,
                            _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                    dstPtrTemp += 8;
                    pTemp[0] = pTemp[1];
                    pTemp[1] = pTemp[2];
                    pTemp[2] = pTemp[3];
                }
                increment_row_ptrs(srcPtrTemp, kernelSize, -16);
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[7];
                for (int k = 0; k < 7; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][7];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 7; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        __m256 pRow[7], pTemp[2];
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_7x7(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_7x7(pRow, &pTemp[1]);
                        blend_permute_min_7x7_host<1, 3, 7, 15, 31, 63>(
                            pTemp, &pResultPln[c], pxMaskPln, blendRegisterOrder);
                    }
                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_store24_f32pln3_to_f16pkd3_avx(dstPtrTemp, pResultPln);

                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 32) * 32;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[7];
                for (int k = 0; k < 7; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[7] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2], srcPtrTemp[3],
                                       srcPtrTemp[4], srcPtrTemp[5], srcPtrTemp[6]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[7], pTemp[5];
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[2]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<7, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_7x7(pRow, &pTemp[3]);
                    pTemp[4] = avx_p0;

                    __m256 pDst[2];
                    blend_permute_min_7x7_host<7, 63, 1, 15, 127, 3>(&pTemp[0], &pDst[0], pxMaskPkd,
                                                                     blendRegisterOrder);
                    blend_permute_min_7x7_host<7, 63, 1, 15, 127, 3>(&pTemp[1], &pDst[1], pxMaskPkd,
                                                                     blendRegisterOrder);

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    increment_row_ptrs(srcPtrTemp, kernelSize, -12);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        }
    } else if (kernelSize == 9) {
        // erode without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++) {
                Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
                for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                    Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                    T* srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                            srcDescPtr->strides.hStride;
                    T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
                    int vectorLoopCount = 0;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize,
                                                        padLength, unpaddedWidth,
                                                        rowKernelLoopLimit);
                    dstPtrTemp += padLength;
#if __AVX2__
                    __m256 pRow[9];
                    if (alignedLength)
                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);

                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                        // compute min of loaded values from 9 rows
                        __m256 pTemp[2], pDst;
                        min_rows_9x9(pRow, &pTemp[0]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);

                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                         rowKernelLoopLimit);
                        min_rows_9x9(pRow, &pTemp[1]);
                        blend_permute_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(
                            pTemp, &pDst, pxMaskPln, blendRegisterOrder);

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128(
                                (__m128i*)dstPtrTemp,
                                _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 32) * 32;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[9] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2],
                                       srcPtrTemp[3], srcPtrTemp[4], srcPtrTemp[5],
                                       srcPtrTemp[6], srcPtrTemp[7], srcPtrTemp[8]};
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                __m256 pRow[9], pTemp[4];
                if (alignedLength) {
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[2]);
                }

                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    // compute min of loaded values from 9 rows
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[3]);

                    __m256 pDst;
                    blend_permute_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(pTemp, &pDst, pxMaskPkd,
                                                                         blendRegisterOrder);
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        _mm_storeu_si128(
                            (__m128i*)dstPtrTemp,
                            _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                    dstPtrTemp += 8;
                    pTemp[0] = pTemp[1];
                    pTemp[1] = pTemp[2];
                    pTemp[2] = pTemp[3];
                }
                increment_row_ptrs(srcPtrTemp, kernelSize, -16);
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        }
        // erode with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be
               computed using raw c code */
            Rpp32u reservedLength = 2 * padLength;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 16) * 16;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrRows[9];
                for (int k = 0; k < 9; k++)
                    srcPtrRows[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrTemp[3][9];
                for (int c = 0; c < 3; c++) {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 9; k++) srcPtrTemp[c][k] = srcPtrRows[k] + channelStride;
                }
                T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                for (int k = 0; k < padLength; k++) {
                    for (int c = 0; c < 3; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                             unpaddedWidth, rowKernelLoopLimit);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                    __m256 pResultPln[3];
                    for (int c = 0; c < 3; c++) {
                        // compute min of loaded values from 9 rows
                        __m256 pRow[9], pTemp[2];
                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_9x9(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp[c],
                                                                         rowKernelLoopLimit);
                        min_rows_9x9(pRow, &pTemp[1]);

                        blend_permute_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(
                            pTemp, &pResultPln[c], pxMaskPln, blendRegisterOrder);
                    }

                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_store24_f32pln3_to_f16pkd3_avx(dstPtrTemp, pResultPln);
                    dstPtrTemp += 24;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                             padLength, unpaddedWidth, rowKernelLoopLimit);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
            }
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
               since (padLength * 3) number of columns from the beginning and end of each row
               will be computed using raw c code */
            Rpp32u reservedLength = (2 * padLength) * 3;
            Rpp32u alignedLength = 0;
            if (bufferLength > reservedLength)
                alignedLength = ((bufferLength - reservedLength) / 40) * 40;
            T* dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[9];
                for (int k = 0; k < 9; k++)
                    srcPtrTemp[k] = srcPtrChannel + std::min(windowRow + k, maxValidRow) *
                                                        srcDescPtr->strides.hStride;
                T* srcPtrInitial[9] = {srcPtrTemp[0], srcPtrTemp[1], srcPtrTemp[2],
                                       srcPtrTemp[3], srcPtrTemp[4], srcPtrTemp[5],
                                       srcPtrTemp[6], srcPtrTemp[7], srcPtrTemp[8]};
                T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                            dstPtrChannels[2] + i * dstDescPtr->strides.hStride};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                    kernelSize, padLength, unpaddedWidth,
                                                    rowKernelLoopLimit);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12) {
                    __m256 pRow[9], pTemp[5];
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[0]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[1]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[2]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[3]);

                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_morphological_load_NxN<9, T, MorphPad_Erode>(pRow, srcPtrTemp,
                                                                     rowKernelLoopLimit);
                    min_rows_9x9(pRow, &pTemp[4]);

                    __m256 pDst[2];
                    blend_permute_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(
                        &pTemp[0], &pDst[0], pxMaskPkd, blendRegisterOrder);
                    blend_permute_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(
                        &pTemp[1], &pDst[1], pxMaskPkd, blendRegisterOrder);

                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                    increment_row_ptrs(srcPtrTemp, kernelSize, -20);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    int channel = vectorLoopCount % 3;
                    erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel],
                                         vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth,
                                         rowKernelLoopLimit, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

// -------------------- Main erode_float_host_tensor function --------------------

template <typename T>
RppStatus erode_float_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                  RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                  RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                  RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    static_assert((std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value),
                  "T must be Rpp32f or Rpp16f");

    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return erode_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr, kernelSize,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);

        // set the required masks array needed for permute operations
#if __AVX2__
    __m256i pxMaskPln[7] = {avx_pxMaskRotate0To1, avx_pxMaskRotate0To2, avx_pxMaskRotate0To3,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To5, avx_pxMaskRotate0To6,
                            avx_pxMaskRotate0To7};
    __m256i pxMaskPkd[7] = {avx_pxMaskRotate0To3, avx_pxMaskRotate0To6, avx_pxMaskRotate0To1,
                            avx_pxMaskRotate0To4, avx_pxMaskRotate0To7, avx_pxMaskRotate0To2,
                            avx_pxMaskRotate0To5};
#endif

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

#if __AVX2__
        erode_float_host_tensor_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, kernelSize,
                                     roi, layoutParams, intraThreads, pxMaskPln, pxMaskPkd);
#else
        erode_float_host_tensor_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, kernelSize,
                                     roi, layoutParams, intraThreads);
#endif
    }

    return RPP_SUCCESS;
}

// -------------------- Per-image implementation for generic erode --------------------

template <typename T>
static inline RppStatus erode_generic_host_tensor_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                       T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                       Rpp32u kernelSize, RpptROI roi,
                                                       RppLayoutParams layoutParams,
                                                       Rpp32u intraThreads) {
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
            T* srcPtrChannelLocal = srcPtrChannel;
            T* dstPtrChannelLocal = dstPtrChannel;

            Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                int vectorLoopCount = 0;
                Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
                T* srcPtrTemp[kernelSize];
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[k] = srcPtrChannelLocal + std::min(windowRow + k, maxValidRow) *
                                                             srcDescPtr->strides.hStride;
                T* dstPtrTemp = dstPtrChannelLocal + i * dstDescPtr->strides.hStride;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength,
                                                    unpaddedWidth, rowKernelLoopLimit);
                dstPtrTemp += padLength;
                vectorLoopCount += padLength;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
            }
            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    } else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++)
                srcPtrTemp[k] = srcPtrChannel +
                                std::min(windowRow + k, maxValidRow) * srcDescPtr->strides.hStride;
            T* srcPtrInitial[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrInitial[k] = srcPtrTemp[k];
            T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrInitial, dstPtrTemp, kernelSize,
                                                padLength, unpaddedWidth, rowKernelLoopLimit);
            dstPtrTemp += padLength * 3;
            vectorLoopCount += padLength * 3;

            // process remaining columns in each row
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize,
                                     padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTemp++;
            }
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
            T* srcPtrTemp[3][kernelSize];
            for (int c = 0; c < 3; c++) {
                Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[c][k] =
                        srcPtrChannel + channelStride +
                        std::min(windowRow + k, maxValidRow) * srcDescPtr->strides.hStride;
            }
            T* dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;
            int vectorLoopCount = 0;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

            // process padLength number of columns in each row
            for (int k = 0; k < padLength; k++) {
                for (int c = 0; c < 3; c++) {
                    erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength,
                                         unpaddedWidth, rowKernelLoopLimit);
                    dstPtrTemp++;
                }
            }
            vectorLoopCount += padLength;

            // process remaining columns in each row
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                for (int c = 0; c < srcDescPtr->c; c++) {
                    erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize,
                                         padLength, unpaddedWidth, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                    dstPtrTemp++;
                }
            }
        }
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        T* dstPtrChannels[3];
        for (int c = 0; c < 3; c++)
            dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
        Rpp32u maxValidRow = roi.xywhROI.roiHeight - 1;
#pragma omp parallel for if (intraThreads > 1) num_threads(intraThreads)
        for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
            Rpp32u windowRow = (i < padLength) ? 0 : (i - padLength);
            T* srcPtrTemp[kernelSize];
            for (int k = 0; k < kernelSize; k++)
                srcPtrTemp[k] = srcPtrChannel +
                                std::min(windowRow + k, maxValidRow) * srcDescPtr->strides.hStride;
            T* srcPtrInitial[kernelSize];
            for (int k = 0; k < kernelSize; k++) srcPtrInitial[k] = srcPtrTemp[k];
            T* dstPtrTempChannels[3] = {dstPtrChannels[0] + i * dstDescPtr->strides.hStride,
                                        dstPtrChannels[1] + i * dstDescPtr->strides.hStride,
                                        dstPtrChannels[2] + i * dstDescPtr->strides.hStride};
            int vectorLoopCount = 0;

            Rpp32s rowKernelLoopLimit = kernelSize;
            get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
            process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrInitial, dstPtrTempChannels,
                                                kernelSize, padLength, unpaddedWidth,
                                                rowKernelLoopLimit);
            vectorLoopCount += padLength * 3;

            // process remaining columns in each row
            for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                int channel = vectorLoopCount % 3;
                erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3,
                                     kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, 3);
                increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                dstPtrTempChannels[channel]++;
            }
        }
    }

    return RPP_SUCCESS;
}

// -------------------- Main erode_generic_host_tensor function --------------------

template <typename T>
RppStatus erode_generic_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                    RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                    RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                    RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);

    omp_set_dynamic(0);
#pragma omp parallel for if (dstDescPtr->n > 1) num_threads(handle.GetNumThreads())
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        Rpp32u intraThreads = get_intra_image_threads(handle, dstDescPtr->n, roi.xywhROI.roiHeight);

        T* srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T* dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        erode_generic_host_tensor_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr, kernelSize,
                                       roi, layoutParams, intraThreads);
    }

    return RPP_SUCCESS;
}

// -------------------- Template instantiations --------------------

template RppStatus erode_char_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr, Rpp32u,
                                                 RpptROIPtr, RpptRoiType, RppLayoutParams,
                                                 rpp::Handle&);

template RppStatus erode_char_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr, Rpp32u,
                                                 RpptROIPtr, RpptRoiType, RppLayoutParams,
                                                 rpp::Handle&);

template RppStatus erode_float_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                   Rpp32u, RpptROIPtr, RpptRoiType, RppLayoutParams,
                                                   rpp::Handle&);

template RppStatus erode_float_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*, RpptDescPtr,
                                                   Rpp32u, RpptROIPtr, RpptRoiType, RppLayoutParams,
                                                   rpp::Handle&);

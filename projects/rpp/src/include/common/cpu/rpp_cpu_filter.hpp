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

#ifndef RPP_CPU_FILTER_HPP
#define RPP_CPU_FILTER_HPP

#include "rpp_cpu_simd_load_store.hpp"

// declare masks used for shuffle and permute operations used in filter functions
const __m128i xmm_pxMaskRotate0To1 =
    _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
const __m128i xmm_pxMaskRotate0To3 =
    _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
const __m128i xmm_pxMaskRotate0To5 =
    _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5);
const __m128i xmm_pxMaskRotate0To7 =
    _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
const __m128i xmm_pxMaskRotate0To9 =
    _mm_setr_epi8(10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
const __m128i xmm_pxMaskRotate0To11 =
    _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
const __m128i xmm_pxMaskRotate0To13 =
    _mm_setr_epi8(14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
const __m256i avx_pxMaskRotate0To1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
const __m256i avx_pxMaskRotate0To2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
const __m256i avx_pxMaskRotate0To3 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);
const __m256i avx_pxMaskRotate0To4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
const __m256i avx_pxMaskRotate0To5 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
const __m256i avx_pxMaskRotate0To6 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
const __m256i avx_pxMaskRotate0To7 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

// increment the kernel size number of row pointers with increment value
template <typename T>
inline void increment_row_ptrs(T** srcPtrTemp, Rpp32u kernelSize, Rpp32s increment) {
    for (int i = 0; i < kernelSize; i++) srcPtrTemp[i] += increment;
}

// get the kernel loop limit based on index
inline void get_kernel_loop_limit(Rpp32s& index, Rpp32s& loopLimit, Rpp32u& padLength,
                                  Rpp32u& unpaddedLength) {
    if ((index < padLength) || (index >= unpaddedLength)) {
        Rpp32u factor = (index < padLength) ? (index - padLength) : (unpaddedLength - 1 - index);
        loopLimit += factor;
    }
}

template <typename T>
inline void convolution_filter_generic_tensor(
    T** srcPtrTemp, T* dstPtrTemp, Rpp32s columnIndex, Rpp32u kernelSize, Rpp32u padLength,
    Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f* filterTensor, Rpp32u channels = 1,
    RpptImageBorderEdge padVertical = RpptImageBorderEdge::BOTTOM_EDGE,
    RpptImageBorderEdge padHorizontal = RpptImageBorderEdge::RIGHT_EDGE) {
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);

    if (rowKernelLoopLimit < kernelSize || columnKernelLoopLimit < kernelSize) {
        for (int i = 0; i < kernelSize; i++) {
            // Compute actual row for vertical clamping
            Rpp32s rowOffset =
                (padVertical == RpptImageBorderEdge::TOP_EDGE)
                    ? std::max(0, static_cast<Rpp32s>(i + rowKernelLoopLimit -
                                                      kernelSize))  // clamp top padded region
                    : std::min(rowKernelLoopLimit - 1, i);          // clamp bottom padded region

            Rpp32s filterRowOffset = i * kernelSize;
            for (int j = 0; j < kernelSize; j++) {
                // Compute actual column for horizontal clamping
                Rpp32s colOffset =
                    (padHorizontal == RpptImageBorderEdge::LEFT_EDGE)
                        ? std::max(0, static_cast<Rpp32s>(j + columnKernelLoopLimit -
                                                          kernelSize))  // clamp left padded region
                        : std::min(static_cast<Rpp32s>(columnKernelLoopLimit - 1),
                                   j);  // clamp right padded region

                // Access and convert pixel
                Rpp32f pixel;
                if constexpr (std::is_same<T, Rpp8s>::value)
                    pixel = static_cast<Rpp32f>(srcPtrTemp[rowOffset][colOffset * channels] + 128);
                else
                    pixel = static_cast<Rpp32f>(srcPtrTemp[rowOffset][colOffset * channels]);

                // Apply filter
                accum = std::fmaf(pixel, filterTensor[filterRowOffset + j], accum);
            }
        }
    } else {
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                Rpp32f pixel;
                if constexpr (std::is_same<T, Rpp8s>::value)
                    pixel = static_cast<Rpp32f>(srcPtrTemp[i][j * channels] + 128);
                else
                    pixel = static_cast<Rpp32f>(srcPtrTemp[i][j * channels]);

                // Apply filter
                accum = std::fmaf(pixel, filterTensor[i * kernelSize + j], accum);
            }
        }
    }

    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        accum = nearbyintf(accum);
    saturate_pixel(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them
// separately
template <typename T>
inline void process_left_border_columns_pln_pln(T** srcPtrTemp, T* dstPtrTemp, Rpp32u kernelSize,
                                                Rpp32u padLength, Rpp32u unpaddedWidth,
                                                Rpp32s rowKernelLoopLimit, Rpp32f* filterTensor,
                                                RpptImageBorderEdge padVertical) {
    for (int k = 0; k < padLength; k++) {
        convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength,
                                          unpaddedWidth, rowKernelLoopLimit, filterTensor, 1,
                                          padVertical, RpptImageBorderEdge::LEFT_EDGE);
        dstPtrTemp++;
    }
}

template <typename T>
inline void process_left_border_columns_pkd_pkd(T** srcPtrTemp, T** srcPtrRow, T* dstPtrTemp,
                                                Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                                Rpp32f* filterTensor,
                                                RpptImageBorderEdge padVertical) {
    for (int c = 0; c < 3; c++) {
        T* dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++) {
            convolution_filter_generic_tensor(
                srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength, unpaddedWidth,
                rowKernelLoopLimit, filterTensor, 3, padVertical, RpptImageBorderEdge::LEFT_EDGE);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }
    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

template <typename T>
inline void process_left_border_columns_pkd_pln(T** srcPtrTemp, T** srcPtrRow,
                                                T** dstPtrTempChannels, Rpp32u kernelSize,
                                                Rpp32u padLength, Rpp32u unpaddedWidth,
                                                Rpp32s rowKernelLoopLimit, Rpp32f* filterTensor,
                                                RpptImageBorderEdge padVertical) {
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < padLength; k++) {
            convolution_filter_generic_tensor(
                srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, unpaddedWidth,
                rowKernelLoopLimit, filterTensor, 3, padVertical, RpptImageBorderEdge::LEFT_EDGE);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++) srcPtrTemp[k] = srcPtrRow[k];
}

// extract 4 SSE registers from 2 AVX registers
inline void extract_4sse_registers(__m256i* pxRowHalf, __m128i* px128) {
    px128[0] = _mm256_castsi256_si128(pxRowHalf[0]);
    px128[1] = _mm256_castsi256_si128(pxRowHalf[1]);
    px128[2] = _mm256_extracti128_si256(pxRowHalf[0], 1);
    px128[3] = _mm256_extracti128_si256(pxRowHalf[1], 1);
}

// extract 3 SSE registers from 2 AVX registers
inline void extract_3sse_registers(__m256i* pxRowHalf, __m128i* px128) {
    px128[0] = _mm256_castsi256_si128(pxRowHalf[0]);
    px128[1] = _mm256_castsi256_si128(pxRowHalf[1]);
    px128[2] = _mm256_extracti128_si256(pxRowHalf[0], 1);
}

// -------------------- U8/I8 bitdepth compute functions for kernel size (3/5/7/9)
// --------------------

// perform required blend shuffle min operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_shuffle_min_3x3_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask
       [0000 0011] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) */
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    px128[0] = _mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[0]), pxTemp[1]);
}

// perform required blend shuffle min operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_shuffle_min_5x5_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08] pxTemp[1] -
       [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] -
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle) pxTemp[2] -
       [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] -
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) pxTemp[3] -
       [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] -
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle) | pxTemp[3] -
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle) */
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    px128[0] = _mm_min_epi16(
        _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
}

// perform required blend shuffle min operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_shuffle_min_7x7_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) */
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    px128[0] =
        _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]);
    px128[0] =
        _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[3]), pxTemp[4]), pxTemp[5]);
}

// perform required blend shuffle min operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_shuffle_min_9x9_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) pxTemp[6] - [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and
       shuffle)    | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) | pxTemp[6] - [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and
       shuffle) */
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[6]], px128[index[6] + 1], blendMask7),
                                 pxMask[6]);
    px128[0] = _mm_min_epi16(
        _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
    px128[0] = _mm_min_epi16(
        _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(px128[0], pxTemp[4]), pxTemp[5]), pxTemp[6]),
        px128[index[6] + 1]);
}

// perform required blend shuffle max operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_shuffle_max_3x3_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask
       [0000 0011] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) */
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    px128[0] = _mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[0]), pxTemp[1]);
}

// perform required blend shuffle max operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_shuffle_max_5x5_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08] pxTemp[1] -
       [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] -
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle) pxTemp[2] -
       [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] -
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) pxTemp[3] -
       [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] -
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle) | pxTemp[3] -
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle) */
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    px128[0] = _mm_max_epi16(
        _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
}

// perform required blend shuffle max operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_shuffle_max_7x7_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) */
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    px128[0] =
        _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]);
    px128[0] =
        _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[3]), pxTemp[4]), pxTemp[5]);
}

// perform required blend shuffle max operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_shuffle_max_9x9_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) pxTemp[6] - [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and
       shuffle)    | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) | pxTemp[6] - [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and
       shuffle) */
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[6]], px128[index[6] + 1], blendMask7),
                                 pxMask[6]);
    px128[0] = _mm_max_epi16(
        _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
    px128[0] = _mm_max_epi16(
        _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(px128[0], pxTemp[4]), pxTemp[5]), pxTemp[6]),
        px128[index[6] + 1]);
}

// perform required blend shuffle add operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_shuffle_add_3x3_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask
       [0000 0011] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) */
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]);
}

// perform required blend shuffle add operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_shuffle_add_5x5_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08] pxTemp[1] -
       [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] -
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle) pxTemp[2] -
       [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] -
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) pxTemp[3] -
       [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] -
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle) | pxTemp[3] -
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle) */
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    px128[0] = _mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
}

// perform required blend shuffle add operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_shuffle_add_7x7_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) */
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    px128[0] =
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]);
    px128[0] =
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[3]), pxTemp[4]), pxTemp[5]);
}

// perform required blend shuffle add operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_shuffle_add_9x9_host(__m128i* px128, __m128i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                             |
       For PKD inputs px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with
       mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with
       mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with
       mask [0000 0111] and shuffle) pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask
       [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask
       [0011 1111] and shuffle) pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000
       1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000
       0001] and shuffle) pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111]
       and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111]
       and shuffle) pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and
       shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and
       shuffle) pxTemp[6] - [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and
       shuffle)    | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and
       shuffle) | pxTemp[6] - [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and
       shuffle) */
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1),
                                 pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2),
                                 pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3),
                                 pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4),
                                 pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5),
                                 pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6),
                                 pxMask[5]);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[6]], px128[index[6] + 1], blendMask7),
                                 pxMask[6]);
    px128[0] = _mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),
        pxTemp[3]);
    px128[0] = _mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[4]), pxTemp[5]), pxTemp[6]),
        px128[index[6] + 1]);
}

// -------------------- F32/F16 bitdepth compute functions for kernel size (3/5/7/9)
// --------------------

// perform required blend permute min multiplication operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_permute_min_3x3_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) */
    pDst[0] = _mm256_min_ps(
        pSrc[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));
    pDst[0] = _mm256_min_ps(
        pDst[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));
}

// perform required blend permute min multiplication operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_permute_min_5x5_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*   For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08]
                   [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                   [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute)
                   [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute) |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute) */
    pDst[0] = _mm256_min_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
}

// perform required blend permute min multiplication operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_permute_min_7x7_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute) |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) */
    pDst[0] = _mm256_min_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
}

// perform required blend permute min multiplication operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_permute_min_9x9_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute)
                  [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and permute)   |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) |
       [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and permute)
    */
    pDst[0] = _mm256_min_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_min_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[6]], pSrc[index[6] + 1], blendMask7),
                                 pxMask[6]));  // blend with mask [0111 1111] and permute -
                                               // [X08|X09|X10|X11|X12|X13|X14|X15]
    pDst[0] = _mm256_min_ps(pDst[0], pSrc[index[6] + 1]);
}

// perform required blend permute max operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_permute_max_3x3_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) */
    pDst[0] = _mm256_max_ps(
        pSrc[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));
    pDst[0] = _mm256_max_ps(
        pDst[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));
}

// perform required blend permute max operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_permute_max_5x5_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*   For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08]
                   [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                   [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute)
                   [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute) |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute) */
    pDst[0] = _mm256_max_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
}

// perform required blend permute max operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_permute_max_7x7_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute) |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) */
    pDst[0] = _mm256_max_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
}

// perform required blend permute max operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_permute_max_9x9_host(__m256* pSrc, __m256* pDst, __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute)
                  [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and permute)   |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) |
       [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and permute)
    */
    pDst[0] = _mm256_max_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_max_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[6]], pSrc[index[6] + 1], blendMask7),
                                 pxMask[6]));  // blend with mask [0111 1111] and permute -
                                               // [X08|X09|X10|X11|X12|X13|X14|X15]
    pDst[0] = _mm256_max_ps(pDst[0], pSrc[index[6] + 1]);
}

// perform required blend permute add multiplication operations for 3x3 kernel size
template <int blendMask1, int blendMask2>
inline void blend_permute_add_mul_3x3_host(__m256* pSrc, __m256* pDst, __m256 pConvolutionFactor,
                                           __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) */
    pDst[0] = _mm256_add_ps(
        pSrc[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));
    pDst[0] = _mm256_add_ps(
        pDst[0], _mm256_permutevar8x32_ps(
                     _mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4>
inline void blend_permute_add_mul_5x5_host(__m256* pSrc, __m256* pDst, __m256 pConvolutionFactor,
                                           __m256i* pxMask, Rpp32u* index) {
    /*   For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06] [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08]
                   [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |
       [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                   [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |
       [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute)
                   [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute) |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute) */
    pDst[0] = _mm256_add_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6>
inline void blend_permute_add_mul_7x7_host(__m256* pSrc, __m256* pDst, __m256 pConvolutionFactor,
                                           __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute) |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) */
    pDst[0] = _mm256_add_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5,
          int blendMask6, int blendMask7>
inline void blend_permute_add_mul_9x9_host(__m256* pSrc, __m256* pDst, __m256 pConvolutionFactor,
                                           __m256i* pxMask, Rpp32u* index) {
    /*  For PLN inputs                                                                          |
       For PKD inputs pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] -
       [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] -
       [B03|R04|G04|B04|R05|G05|B05|R06], [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000
       0001] and permute)     pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] -
       [R09|G09|B09|R10|G10|B10|R11|G11] [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000
       0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000
       0111] and permute) [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and
       permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and
       permute) [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |
       [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |
       [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |
       [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute)
                  [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and permute)   |
       [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) |
       [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and permute)
    */
    pDst[0] = _mm256_add_ps(
        pSrc[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1),
                                 pxMask[0]));  // blend with mask [0000 0001] and permute -
                                               // [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2),
                                 pxMask[1]));  // blend with mask [0000 0011] and permute -
                                               // [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3),
                                 pxMask[2]));  // blend with mask [0000 0111] and permute -
                                               // [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4),
                                 pxMask[3]));  // blend with mask [0000 1111] and permute -
                                               // [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5),
                                 pxMask[4]));  // blend with mask [0001 1111] and permute -
                                               // [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6),
                                 pxMask[5]));  // blend with mask [0011 1111] and permute -
                                               // [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_add_ps(
        pDst[0],
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[6]], pSrc[index[6] + 1], blendMask7),
                                 pxMask[6]));  // blend with mask [0111 1111] and permute -
                                               // [X08|X09|X10|X11|X12|X13|X14|X15]
    pDst[0] = _mm256_add_ps(pDst[0], pSrc[index[6] + 1]);
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

template <int blendMask1, int blendMask2, int roatateMask1, int roatateMask2>
inline void permute_blend_add_3x3(__m256& pDst, __m256 pRow0, __m256 pRow1, __m256* pFilter,
                                  __m256i* pxMask) {
    pDst = _mm256_fmadd_ps(pRow0, pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, blendMask1), pxMask[roatateMask1]),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, blendMask2), pxMask[roatateMask2]),
        pFilter[2], pDst);
}

inline void permute_blend_add_5x5_pln(__m256& pDst, __m256 pRow0, __m256 pRow1, __m256* pFilter,
                                      bool debug = false) {
    pDst = _mm256_fmadd_ps(pRow0, pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, 1), avx_pxMaskRotate0To1),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, 3), avx_pxMaskRotate0To2),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, 7), avx_pxMaskRotate0To3),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow0, pRow1, 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
}

inline void permute_blend_add_5x5_pkd(__m256& pDst, __m256* pRow, __m256* pFilter) {
    pDst = _mm256_fmadd_ps(pRow[0], pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 7), avx_pxMaskRotate0To3),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 63), avx_pxMaskRotate0To6),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 1), avx_pxMaskRotate0To1),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
}

inline void permute_blend_add_7x7_pln(__m256& pDst, __m256* pRow, __m256* pFilter) {
    pDst = _mm256_fmadd_ps(pRow[0], pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 1), avx_pxMaskRotate0To1),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 3), avx_pxMaskRotate0To2),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 7), avx_pxMaskRotate0To3),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 31), avx_pxMaskRotate0To5),
        pFilter[5], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 63), avx_pxMaskRotate0To6),
        pFilter[6], pDst);
}

inline void permute_blend_add_7x7_pkd(__m256& pDst, __m256* pRow, __m256 pRow2, __m256* pFilter) {
    pDst = _mm256_fmadd_ps(pRow[0], pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 7), avx_pxMaskRotate0To3),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 63), avx_pxMaskRotate0To6),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 1), avx_pxMaskRotate0To1),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 127), avx_pxMaskRotate0To7),
        pFilter[5], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[2], pRow2, 3), avx_pxMaskRotate0To2),
        pFilter[6], pDst);
}

inline void permute_blend_add_9x9_pln(__m256& pDst, __m256* pRow, __m256* pFilter) {
    pDst = _mm256_fmadd_ps(pRow[0], pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 1), avx_pxMaskRotate0To1),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 3), avx_pxMaskRotate0To2),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 7), avx_pxMaskRotate0To3),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 31), avx_pxMaskRotate0To5),
        pFilter[5], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 63), avx_pxMaskRotate0To6),
        pFilter[6], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 127), avx_pxMaskRotate0To7),
        pFilter[7], pDst);
    pDst = _mm256_fmadd_ps(pRow[1], pFilter[8], pDst);
}

inline void permute_blend_add_9x9_pkd(__m256& pDst, __m256* pRow, __m256* pFilter) {
    pDst = _mm256_fmadd_ps(pRow[0], pFilter[0], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 7), avx_pxMaskRotate0To3),
        pFilter[1], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[0], pRow[1], 63), avx_pxMaskRotate0To6),
        pFilter[2], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 1), avx_pxMaskRotate0To1),
        pFilter[3], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 15), avx_pxMaskRotate0To4),
        pFilter[4], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[1], pRow[2], 127), avx_pxMaskRotate0To7),
        pFilter[5], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[2], pRow[3], 3), avx_pxMaskRotate0To2),
        pFilter[6], pDst);
    pDst = _mm256_fmadd_ps(
        _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[2], pRow[3], 31), avx_pxMaskRotate0To5),
        pFilter[7], pDst);
    pDst = _mm256_fmadd_ps(pRow[3], pFilter[8], pDst);
}

// -------------------- Filter load functions for NxN kernels - 3x3, 5x5, 7x7 and 9x9
// --------------------

template <int FILTER_SIZE, typename T>
inline void rpp_load_filter_NxN_pln_host(__m256* pRow, T** srcPtrTemp, Rpp32s rowKernelLoopLimit,
                                         Rpp32s padIndex) {
    // Determine the starting row offset based on whether padding is applied
    // If padIndex is non-zero (true), start from the radius position of the kernel; otherwise start
    // from 0
    const int radius = FILTER_SIZE - rowKernelLoopLimit;
    int centerRowOffset = padIndex ? radius : 0;

#pragma unroll
    for (int k = 0; k < FILTER_SIZE; k++) {
        // Calculate the desired row index relative to the calculated centerRowOffset of the filter
        // This maps filter positions to actual row indices in the source data
        int desiredIndex = centerRowOffset + (k - radius);
        int clampedIndex = std::max(0, std::min(desiredIndex, rowKernelLoopLimit - 1));

        if constexpr (std::is_same_v<T, Rpp8s>)
            rpp_load16_i8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 2]);
        else if constexpr (std::is_same_v<T, Rpp8u>)
            rpp_load16_u8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 2]);
        else if constexpr (std::is_same_v<T, Rpp16f>)
            rpp_load16_f16_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 2]);
        else if constexpr (std::is_same_v<T, Rpp32f>)
            rpp_load16_f32_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 2]);
    }
}

template <int FILTER_SIZE, typename T>
inline void rpp_load_filter_NxN_pkd_host(__m256* pRow, T** srcPtrTemp, Rpp32s rowKernelLoopLimit,
                                         Rpp32s padIndex) {
    // Determine the starting row offset based on whether padding is applied
    // If padIndex is non-zero (true), start from the radius position of the kernel; otherwise start
    // from 0
    const int radius = FILTER_SIZE - rowKernelLoopLimit;
    int centerRowOffset = padIndex ? radius : 0;

#pragma unroll
    for (int k = 0; k < FILTER_SIZE; k++) {
        // Calculate the desired row index relative to the calculated centerRowOffset of the filter
        // This maps filter positions to actual row indices in the source data
        int desiredIndex = centerRowOffset + (k - radius);
        int clampedIndex = std::max(0, std::min(desiredIndex, rowKernelLoopLimit - 1));

        if constexpr (std::is_same_v<T, Rpp8u>)
            rpp_load32_u8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 4]);
        else if constexpr (std::is_same_v<T, Rpp8s>)
            rpp_load32_i8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 4]);
        else if constexpr (std::is_same_v<T, Rpp32f>)
            rpp_load32_f32_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 4]);
        else if constexpr (std::is_same_v<T, Rpp16f>)
            rpp_load32_f16_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 4]);
    }
}

// Specialized load function for 9x9 PKD3 -> PLN3 case - Loads 40 pixels per row
template <typename T>
inline void rpp_load_gaussian_filter_9x9_pkd_pln_host(__m256* pRow, T** srcPtrTemp,
                                                      Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // Determine the starting row offset based on whether padding is applied
    // If padIndex is non-zero (true), start from the radius position of the kernel; otherwise start
    // from 0
    const int radius = 9 - rowKernelLoopLimit;
    int centerRowOffset = padIndex ? radius : 0;

#pragma unroll
    for (int k = 0; k < 9; k++) {
        // Calculate the desired row index relative to the calculated centerRowOffset of the filter
        // This maps filter positions to actual row indices in the source data
        int desiredIndex = centerRowOffset + (k - radius);
        int clampedIndex = std::max(0, std::min(desiredIndex, rowKernelLoopLimit - 1));

        if constexpr (std::is_same_v<T, Rpp8u>)
            rpp_load40_u8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 5]);
        else if constexpr (std::is_same_v<T, Rpp8s>)
            rpp_load40_i8_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 5]);
        else if constexpr (std::is_same_v<T, Rpp32f>)
            rpp_load40_f32_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 5]);
        else if constexpr (std::is_same_v<T, Rpp16f>)
            rpp_load40_f16_to_f32_avx(srcPtrTemp[clampedIndex], &pRow[k * 5]);
    }
}

// -------------------- Filter load functions for U8/I8 bitdepth --------------------

// load function for 3x3 kernel size
template <typename T>
inline void rpp_load_box_filter_char_3x3_host(__m256i* pxRow, T** srcPtrTemp,
                                              Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i*)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i*)srcPtrTemp[1]);
    if (rowKernelLoopLimit == 3)
        pxRow[2] = _mm256_loadu_si256((__m256i*)srcPtrTemp[2]);
    else
        pxRow[2] = pxRow[padIndex];
}

// load function for 5x5 kernel size
template <typename T>
inline void rpp_load_box_filter_char_5x5_host(__m256i* pxRow, T** srcPtrTemp,
                                              Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i*)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i*)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i*)srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i*)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++) pxRow[k] = pxRow[padIndex];
}

// load function for 7x7 kernel size
template <typename T>
inline void rpp_load_box_filter_char_7x7_host(__m256i* pxRow, T** srcPtrTemp,
                                              Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i*)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i*)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i*)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i*)srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i*)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++) pxRow[k] = pxRow[padIndex];
}

// load function for 9x9 kernel size
template <typename T>
inline void rpp_load_box_filter_char_9x9_host(__m256i* pxRow, T** srcPtrTemp,
                                              Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i*)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i*)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i*)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i*)srcPtrTemp[3]);
    pxRow[4] = _mm256_loadu_si256((__m256i*)srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i*)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++) pxRow[k] = pxRow[padIndex];
}

// -------------------- Filter load functions for F32 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_float_3x3_host(__m256* pRow, Rpp32f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    else
        pRow[2] = pRow[padIndex];
}

// load function for 5x5 kernel size
inline void rpp_load_box_filter_float_5x5_host(__m256* pRow, Rpp32f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++) pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++) pRow[k] = pRow[padIndex];
}

// load function for 7x7 kernel size
inline void rpp_load_box_filter_float_7x7_host(__m256* pRow, Rpp32f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++) pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++) pRow[k] = pRow[padIndex];
}

// load function for 9x9 kernel size
inline void rpp_load_box_filter_float_9x9_host(__m256* pRow, Rpp32f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    pRow[4] = _mm256_loadu_ps(srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++) pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++) pRow[k] = pRow[padIndex];
}

// -------------------- Filter load functions for F16 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_float_3x3_host(__m256* pRow, Rpp16f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[0]))));
    pRow[1] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[1]))));
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_cvtph_ps(
            _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[2]))));
    else
        pRow[2] = pRow[padIndex];
}

// load function for 5x5 kernel size
inline void rpp_load_box_filter_float_5x5_host(__m256* pRow, Rpp16f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pRow[0] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[0]))));
    pRow[1] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[1]))));
    pRow[2] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[2]))));
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(
            _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 5; k++) pRow[k] = pRow[padIndex];
}

// load function for 7x7 kernel size
inline void rpp_load_box_filter_float_7x7_host(__m256* pRow, Rpp16f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pRow[0] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[0]))));
    pRow[1] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[1]))));
    pRow[2] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[2]))));
    pRow[3] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[3]))));
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(
            _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 7; k++) pRow[k] = pRow[padIndex];
}

// load function for 9x9 kernel size
inline void rpp_load_box_filter_float_9x9_host(__m256* pRow, Rpp16f** srcPtrTemp,
                                               Rpp32s rowKernelLoopLimit, Rpp32s padIndex) {
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pRow[0] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[0]))));
    pRow[1] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[1]))));
    pRow[2] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[2]))));
    pRow[3] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[3]))));
    pRow[4] =
        _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[4]))));
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(
            _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f*>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 9; k++) pRow[k] = pRow[padIndex];
}

template <typename T>
struct MorphVecLoader;

template <>
struct MorphVecLoader<Rpp8u> {
    using VecType = __m256i;
    static inline VecType load(void* ptr) {
        return _mm256_loadu_si256((__m256i*)ptr);
    }
};

template <>
struct MorphVecLoader<Rpp8s> {
    using VecType = __m256i;
    static inline VecType load(void* ptr) {
        return _mm256_loadu_si256((__m256i*)ptr);
    }
};

template <>
struct MorphVecLoader<Rpp32f> {
    using VecType = __m256;
    static inline VecType load(void* ptr) {
        return _mm256_loadu_ps((float*)ptr);
    }
};

template <>
struct MorphVecLoader<Rpp16f> {
    using VecType = __m256;
    static inline VecType load(void* ptr) {
        return _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps((float*)ptr)));
    }
};

struct MorphPad_Erode {
    static inline __m256i pad_int() {
        return _mm256_set1_epi8((char)255);
    }
    static inline __m256 pad_float() {
        return avx_p1;
    }
};

struct MorphPad_Dilate {
    static inline __m256i pad_int() {
        return _mm256_set1_epi8((char)0);
    }
    static inline __m256 pad_float() {
        return avx_p0;
    }
};

template <int kernelSize, typename T, typename padPolicy>
inline void rpp_morphological_load_NxN(typename MorphVecLoader<T>::VecType* pxRow, T** srcPtrTemp,
                                       Rpp32s rowKernelLoopLimit) {
    using Loader = MorphVecLoader<T>;
    using Vec = typename Loader::VecType;

    constexpr int preLoadRows = (kernelSize + 1) / 2;

    // Load initial rows
    for (int k = 0; k < preLoadRows; ++k) pxRow[k] = Loader::load(srcPtrTemp[k]);

    // Load valid remaining rows
    for (int k = preLoadRows; k < rowKernelLoopLimit; ++k) pxRow[k] = Loader::load(srcPtrTemp[k]);

    // Pad beyond valid range
    for (int k = rowKernelLoopLimit; k < kernelSize; ++k) {
        if constexpr (std::is_same_v<Vec, __m256i>)
            pxRow[k] = padPolicy::pad_int();
        else
            pxRow[k] = padPolicy::pad_float();
    }
}

#endif  // RPP_CPU_FILTER_HPP

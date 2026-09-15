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

#include <type_traits>

#include "host_tensor_executors.hpp"
#if __AVX2__
#include <immintrin.h>
#endif

/* median filter algorithm explanation

Algorithm Selection (based on kernel size):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Small kernels (3×3, 5×5): Sorting networks + AVX2 SIMD vectorization
   - Uses hardcoded optimal compare-swap sequences
   - Processes 8-32 elements simultaneously with SIMD
   - No branching in inner loops

2. Large kernels (7×7, 9×9+): Histogram-based method (U8 only)
   - Single-level 256-bin histogram over the current kernel window
   - Histogram is rebuilt from scratch for each output pixel (no sliding window)
   - Median found by cumulative count scan; cost grows with kernel area

3. Fallback: Generic std::nth_element (for non-U8 types with large kernels)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example for 3×3 U8 PLN1:

Input (3×32 grayscale image):
x  x  x  x  x  x  x  x  x  x  ..  x  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  x  x  x  x  x  x  x  x  x  ..  x  x

For each output pixel:
1. Load 3×3 window with border replication (nearest-neighbor padding)
2. Apply sorting network (19 compare-swap operations)
3. Extract median value (element at position 4 after sorting)
4. Write to output

AVX2 optimization processes 32 pixels simultaneously using vector min/max operations.
*/

template <typename T>
inline void rpp_minmax_op_scalar(T& a, T& b) {
    T t = a;
    a = std::min(a, b);
    b = std::max(b, t);
}

// 3x3 sorting network. After ops, p[4] is median.
template <typename T>
inline T rpp_median_3x3_sortnet(T* p) {
#define op(i, j) rpp_minmax_op_scalar(p[i], p[j])
    op(1, 2);
    op(4, 5);
    op(7, 8);
    op(0, 1);
    op(3, 4);
    op(6, 7);
    op(1, 2);
    op(4, 5);
    op(7, 8);
    op(0, 3);
    op(5, 8);
    op(4, 7);
    op(3, 6);
    op(1, 4);
    op(2, 5);
    op(4, 7);
    op(4, 2);
    op(6, 4);
    op(4, 2);
#undef op
    return p[4];
}

// 5x5 sorting network. After ops, p[12] is median.
template <typename T>
inline T rpp_median_5x5_sortnet(T* p) {
#define op(i, j) rpp_minmax_op_scalar(p[i], p[j])
    op(1, 2);
    op(0, 1);
    op(1, 2);
    op(4, 5);
    op(3, 4);
    op(4, 5);
    op(0, 3);
    op(2, 5);
    op(2, 3);
    op(1, 4);
    op(1, 2);
    op(3, 4);
    op(7, 8);
    op(6, 7);
    op(7, 8);
    op(10, 11);
    op(9, 10);
    op(10, 11);
    op(6, 9);
    op(8, 11);
    op(8, 9);
    op(7, 10);
    op(7, 8);
    op(9, 10);
    op(0, 6);
    op(4, 10);
    op(4, 6);
    op(2, 8);
    op(2, 4);
    op(6, 8);
    op(1, 7);
    op(5, 11);
    op(5, 7);
    op(3, 9);
    op(3, 5);
    op(7, 9);
    op(1, 2);
    op(3, 4);
    op(5, 6);
    op(7, 8);
    op(9, 10);
    op(13, 14);
    op(12, 13);
    op(13, 14);
    op(16, 17);
    op(15, 16);
    op(16, 17);
    op(12, 15);
    op(14, 17);
    op(14, 15);
    op(13, 16);
    op(13, 14);
    op(15, 16);
    op(19, 20);
    op(18, 19);
    op(19, 20);
    op(21, 22);
    op(23, 24);
    op(21, 23);
    op(22, 24);
    op(22, 23);
    op(18, 21);
    op(20, 23);
    op(20, 21);
    op(19, 22);
    op(22, 24);
    op(19, 20);
    op(21, 22);
    op(23, 24);
    op(12, 18);
    op(16, 22);
    op(16, 18);
    op(14, 20);
    op(20, 24);
    op(14, 16);
    op(18, 20);
    op(22, 24);
    op(13, 19);
    op(17, 23);
    op(17, 19);
    op(15, 21);
    op(15, 17);
    op(19, 21);
    op(13, 14);
    op(15, 16);
    op(17, 18);
    op(19, 20);
    op(21, 22);
    op(23, 24);
    op(0, 12);
    op(8, 20);
    op(8, 12);
    op(4, 16);
    op(16, 24);
    op(12, 16);
    op(2, 14);
    op(10, 22);
    op(10, 14);
    op(6, 18);
    op(6, 10);
    op(10, 12);
    op(1, 13);
    op(9, 21);
    op(9, 13);
    op(5, 17);
    op(13, 17);
    op(3, 15);
    op(11, 23);
    op(11, 15);
    op(7, 19);
    op(7, 11);
    op(11, 13);
    op(11, 12);
#undef op
    return p[12];
}

// Simplified O(1) histogram for single-channel PLN1
inline void rpp_median_histogram_u8_pln1_host(const Rpp8u* src, Rpp8u* dst, int width, int height,
                                              int srcStride, int dstStride, int ksize) {
    int padLength = ksize / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Build histogram for this pixel's window
            int hist[256] = {0};

            for (int dy = -padLength; dy <= padLength; dy++) {
                int row = std::max(0, std::min(i + dy, height - 1));
                for (int dx = -padLength; dx <= padLength; dx++) {
                    int col = std::max(0, std::min(j + dx, width - 1));
                    hist[src[row * srcStride + col]]++;
                }
            }

            // Find median value
            int count = 0;
            int threshold = (ksize * ksize) / 2;
            for (int v = 0; v < 256; v++) {
                count += hist[v];
                if (count > threshold) {
                    dst[i * dstStride + j] = (Rpp8u)v;
                    break;
                }
            }
        }
    }
}

// Simplified histogram for 3-channel PKD
inline void rpp_median_histogram_u8_pkd_host(const Rpp8u* src, Rpp8u* dst, int width, int height,
                                             int srcStride, int dstStride, int ksize,
                                             int channels) {
    int padLength = ksize / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Process each channel separately
            for (int c = 0; c < channels; c++) {
                // Build histogram for this pixel's window for current channel
                int hist[256] = {0};

                for (int dy = -padLength; dy <= padLength; dy++) {
                    int row = std::max(0, std::min(i + dy, height - 1));
                    for (int dx = -padLength; dx <= padLength; dx++) {
                        int col = std::max(0, std::min(j + dx, width - 1));
                        hist[src[row * srcStride + col * channels + c]]++;
                    }
                }

                // Find median value
                int count = 0;
                int threshold = (ksize * ksize) / 2;
                for (int v = 0; v < 256; v++) {
                    count += hist[v];
                    if (count > threshold) {
                        dst[i * dstStride + j * channels + c] = (Rpp8u)v;
                        break;
                    }
                }
            }
        }
    }
}

#if __AVX2__

// -------------------- AVX2 Sorting Network Implementations --------------------

// 3×3 Median - U8 PLN1
inline void rpp_median3x3_pln_u8_avx(const Rpp8u* row0, const Rpp8u* row1, const Rpp8u* row2,
                                     Rpp8u* dstRow, int width) {
    int j = 0;
    const int nlanes = 32;

    // Scalar left edge (first pixel)
    {
        int p0 = row0[0], p1 = row0[0], p2 = row0[std::min(1, width - 1)];
        int p3 = row1[0], p4 = row1[0], p5 = row1[std::min(1, width - 1)];
        int p6 = row2[0], p7 = row2[0], p8 = row2[std::min(1, width - 1)];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[0] = (Rpp8u)rpp_median_3x3_sortnet(p);
        j = 1;
    }

    // Vector path for interior pixels
    for (; j < width - 1; j += nlanes) {
        // Handle tail
        if (j > width - 1 - nlanes) {
            // Exit vectorized loop for small widths or in-place operation to avoid reading
            // overwritten data
            if (j == 1 || (const Rpp8u*)dstRow == row1) break;
            j = width - 1 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - 1));
        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j + 1));
        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row1 + j - 1));
        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j + 1));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row2 + j - 1));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row2 + j + 1));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p1);
        OP(p3, p4);
        OP(p6, p7);
        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p3);
        OP(p5, p8);
        OP(p4, p7);
        OP(p3, p6);
        OP(p1, p4);
        OP(p2, p5);
        OP(p4, p7);
        OP(p4, p2);
        OP(p6, p4);
        OP(p4, p2);

#undef OP

        _mm256_storeu_si256((__m256i*)(dstRow + j), p4);
    }

    // Scalar tail/right edge
    for (; j < width; j++) {
        int j0 = std::max(j - 1, 0);
        int j2 = std::min(j + 1, width - 1);

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8u)rpp_median_3x3_sortnet(p);
    }
}

// 3×3 Median - I8 PLN1 (processes 32 pixels per iteration)
inline void rpp_median3x3_pln_i8_avx(const Rpp8s* row0, const Rpp8s* row1, const Rpp8s* row2,
                                     Rpp8s* dstRow, int width) {
    int j = 0;
    const int nlanes = 32;

    // Scalar left edge (first pixel)
    {
        int p0 = row0[0], p1 = row0[0], p2 = row0[std::min(1, width - 1)];
        int p3 = row1[0], p4 = row1[0], p5 = row1[std::min(1, width - 1)];
        int p6 = row2[0], p7 = row2[0], p8 = row2[std::min(1, width - 1)];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[0] = (Rpp8s)rpp_median_3x3_sortnet(p);
        j = 1;
    }

    // Vector path for interior pixels
    for (; j < width - 1; j += nlanes) {
        if (j > width - 1 - nlanes) {
            if (j == 1 || (const Rpp8s*)dstRow == row1) break;
            j = width - 1 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - 1));
        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j + 1));
        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row1 + j - 1));
        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j + 1));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row2 + j - 1));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row2 + j + 1));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epi8(a, b); \
        b = _mm256_max_epi8(b, t); \
    }

        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p1);
        OP(p3, p4);
        OP(p6, p7);
        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p3);
        OP(p5, p8);
        OP(p4, p7);
        OP(p3, p6);
        OP(p1, p4);
        OP(p2, p5);
        OP(p4, p7);
        OP(p4, p2);
        OP(p6, p4);
        OP(p4, p2);

#undef OP

        _mm256_storeu_si256((__m256i*)(dstRow + j), p4);
    }

    // Scalar tail/right edge
    for (; j < width; j++) {
        int j0 = std::max(j - 1, 0);
        int j2 = std::min(j + 1, width - 1);

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8s)rpp_median_3x3_sortnet(p);
    }
}

// 3×3 Median - F32 PLN1 (processes 8 pixels per iteration)
inline void rpp_median3x3_pln_f32_avx(const Rpp32f* row0, const Rpp32f* row1, const Rpp32f* row2,
                                      Rpp32f* dstRow, int width) {
    int j = 0;
    const int nlanes = 8;

    // Scalar left edge (first pixel)
    {
        float p0 = row0[0], p1 = row0[0], p2 = row0[std::min(1, width - 1)];
        float p3 = row1[0], p4 = row1[0], p5 = row1[std::min(1, width - 1)];
        float p6 = row2[0], p7 = row2[0], p8 = row2[std::min(1, width - 1)];

        float p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[0] = rpp_median_3x3_sortnet(p);
        j = 1;
    }

    // Vector path for interior pixels
    for (; j < width - 1; j += nlanes) {
        if (j > width - 1 - nlanes) {
            if (j == 1 || (const Rpp32f*)dstRow == row1) break;
            j = width - 1 - nlanes;
        }

        __m256 p0 = _mm256_loadu_ps(row0 + j - 1);
        __m256 p1 = _mm256_loadu_ps(row0 + j);
        __m256 p2 = _mm256_loadu_ps(row0 + j + 1);
        __m256 p3 = _mm256_loadu_ps(row1 + j - 1);
        __m256 p4 = _mm256_loadu_ps(row1 + j);
        __m256 p5 = _mm256_loadu_ps(row1 + j + 1);
        __m256 p6 = _mm256_loadu_ps(row2 + j - 1);
        __m256 p7 = _mm256_loadu_ps(row2 + j);
        __m256 p8 = _mm256_loadu_ps(row2 + j + 1);

#define OP(a, b)                 \
    {                            \
        __m256 t = a;            \
        a = _mm256_min_ps(a, b); \
        b = _mm256_max_ps(b, t); \
    }

        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p1);
        OP(p3, p4);
        OP(p6, p7);
        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p3);
        OP(p5, p8);
        OP(p4, p7);
        OP(p3, p6);
        OP(p1, p4);
        OP(p2, p5);
        OP(p4, p7);
        OP(p4, p2);
        OP(p6, p4);
        OP(p4, p2);

#undef OP

        _mm256_storeu_ps(dstRow + j, p4);
    }

    // Scalar tail/right edge
    for (; j < width; j++) {
        int j0 = std::max(j - 1, 0);
        int j2 = std::min(j + 1, width - 1);

        float p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        float p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        float p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        float p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = rpp_median_3x3_sortnet(p);
    }
}

// 3×3 Median - F16 PLN1 (converts to F32, processes 8 pixels, converts back)
inline void rpp_median3x3_pln_f16_avx(const Rpp16f* row0, const Rpp16f* row1, const Rpp16f* row2,
                                      Rpp16f* dstRow, int width) {
    int j = 0;
    const int nlanes = 8;

    // Scalar left edge (first pixel)
    {
        float p0 = row0[0], p1 = row0[0], p2 = row0[std::min(1, width - 1)];
        float p3 = row1[0], p4 = row1[0], p5 = row1[std::min(1, width - 1)];
        float p6 = row2[0], p7 = row2[0], p8 = row2[std::min(1, width - 1)];

        float p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[0] = rpp_median_3x3_sortnet(p);
        j = 1;
    }

    // Vector path for interior pixels
    for (; j < width - 1; j += nlanes) {
        if (j > width - 1 - nlanes) {
            if (j == 1 || (const Rpp16f*)dstRow == row1) break;
            j = width - 1 - nlanes;
        }

        // Load F16 and convert to F32
        __m256 p0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j - 1)));
        __m256 p1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j)));
        __m256 p2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j + 1)));
        __m256 p3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j - 1)));
        __m256 p4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j)));
        __m256 p5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j + 1)));
        __m256 p6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j - 1)));
        __m256 p7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j)));
        __m256 p8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j + 1)));

#define OP(a, b)                 \
    {                            \
        __m256 t = a;            \
        a = _mm256_min_ps(a, b); \
        b = _mm256_max_ps(b, t); \
    }

        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p1);
        OP(p3, p4);
        OP(p6, p7);
        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p3);
        OP(p5, p8);
        OP(p4, p7);
        OP(p3, p6);
        OP(p1, p4);
        OP(p2, p5);
        OP(p4, p7);
        OP(p4, p2);
        OP(p6, p4);
        OP(p4, p2);

#undef OP

        // Convert back to F16 and store
        _mm_storeu_si128((__m128i*)(dstRow + j), _mm256_cvtps_ph(p4, 0));
    }

    // Scalar tail/right edge
    for (; j < width; j++) {
        int j0 = std::max(j - 1, 0);
        int j2 = std::min(j + 1, width - 1);

        float p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        float p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        float p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        float p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = rpp_median_3x3_sortnet(p);
    }
}

// 5×5 Median - U8 PLN1 (processes 32 pixels per iteration)
inline void rpp_median5x5_pln_u8_avx(const Rpp8u* row0, const Rpp8u* row1, const Rpp8u* row2,
                                     const Rpp8u* row3, const Rpp8u* row4, Rpp8u* dstRow,
                                     int width) {
    int j = 0;
    const int nlanes = 32;

    // Scalar left edge (first 2 pixels)
    for (j = 0; j < std::min(2, width); j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }

    // Vector path for interior pixels
    for (; j < width - 2; j += nlanes) {
        if (j > width - 2 - nlanes) {
            if (j == 2 || (const Rpp8u*)dstRow == row2) break;
            j = width - 2 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - 2));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j - 2));
        __m256i p10 = _mm256_loadu_si256((const __m256i*)(row2 + j - 2));
        __m256i p15 = _mm256_loadu_si256((const __m256i*)(row3 + j - 2));
        __m256i p20 = _mm256_loadu_si256((const __m256i*)(row4 + j - 2));

        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j - 1));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row1 + j - 1));
        __m256i p11 = _mm256_loadu_si256((const __m256i*)(row2 + j - 1));
        __m256i p16 = _mm256_loadu_si256((const __m256i*)(row3 + j - 1));
        __m256i p21 = _mm256_loadu_si256((const __m256i*)(row4 + j - 1));

        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p12 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p17 = _mm256_loadu_si256((const __m256i*)(row3 + j));
        __m256i p22 = _mm256_loadu_si256((const __m256i*)(row4 + j));

        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row0 + j + 1));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row1 + j + 1));
        __m256i p13 = _mm256_loadu_si256((const __m256i*)(row2 + j + 1));
        __m256i p18 = _mm256_loadu_si256((const __m256i*)(row3 + j + 1));
        __m256i p23 = _mm256_loadu_si256((const __m256i*)(row4 + j + 1));

        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row0 + j + 2));
        __m256i p9 = _mm256_loadu_si256((const __m256i*)(row1 + j + 2));
        __m256i p14 = _mm256_loadu_si256((const __m256i*)(row2 + j + 2));
        __m256i p19 = _mm256_loadu_si256((const __m256i*)(row3 + j + 2));
        __m256i p24 = _mm256_loadu_si256((const __m256i*)(row4 + j + 2));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        OP(p1, p2);
        OP(p0, p1);
        OP(p1, p2);
        OP(p4, p5);
        OP(p3, p4);
        OP(p4, p5);
        OP(p0, p3);
        OP(p2, p5);
        OP(p2, p3);
        OP(p1, p4);
        OP(p1, p2);
        OP(p3, p4);
        OP(p7, p8);
        OP(p6, p7);
        OP(p7, p8);
        OP(p10, p11);
        OP(p9, p10);
        OP(p10, p11);
        OP(p6, p9);
        OP(p8, p11);
        OP(p8, p9);
        OP(p7, p10);
        OP(p7, p8);
        OP(p9, p10);
        OP(p0, p6);
        OP(p4, p10);
        OP(p4, p6);
        OP(p2, p8);
        OP(p2, p4);
        OP(p6, p8);
        OP(p1, p7);
        OP(p5, p11);
        OP(p5, p7);
        OP(p3, p9);
        OP(p3, p5);
        OP(p7, p9);
        OP(p1, p2);
        OP(p3, p4);
        OP(p5, p6);
        OP(p7, p8);
        OP(p9, p10);
        OP(p13, p14);
        OP(p12, p13);
        OP(p13, p14);
        OP(p16, p17);
        OP(p15, p16);
        OP(p16, p17);
        OP(p12, p15);
        OP(p14, p17);
        OP(p14, p15);
        OP(p13, p16);
        OP(p13, p14);
        OP(p15, p16);
        OP(p19, p20);
        OP(p18, p19);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p21, p23);
        OP(p22, p24);
        OP(p22, p23);
        OP(p18, p21);
        OP(p20, p23);
        OP(p20, p21);
        OP(p19, p22);
        OP(p22, p24);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p12, p18);
        OP(p16, p22);
        OP(p16, p18);
        OP(p14, p20);
        OP(p20, p24);
        OP(p14, p16);
        OP(p18, p20);
        OP(p22, p24);
        OP(p13, p19);
        OP(p17, p23);
        OP(p17, p19);
        OP(p15, p21);
        OP(p15, p17);
        OP(p19, p21);
        OP(p13, p14);
        OP(p15, p16);
        OP(p17, p18);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p0, p12);
        OP(p8, p20);
        OP(p8, p12);
        OP(p4, p16);
        OP(p16, p24);
        OP(p12, p16);
        OP(p2, p14);
        OP(p10, p22);
        OP(p10, p14);
        OP(p6, p18);
        OP(p6, p10);
        OP(p10, p12);
        OP(p1, p13);
        OP(p9, p21);
        OP(p9, p13);
        OP(p5, p17);
        OP(p13, p17);
        OP(p3, p15);
        OP(p11, p23);
        OP(p11, p15);
        OP(p7, p19);
        OP(p7, p11);
        OP(p11, p13);
        OP(p11, p12);

#undef OP

        _mm256_storeu_si256((__m256i*)(dstRow + j), p12);
    }

    // Scalar tail/right edge
    for (; j < width; j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }
}

// 5×5 Median - I8 PLN1 (processes 32 pixels per iteration)
inline void rpp_median5x5_pln_i8_avx(const Rpp8s* row0, const Rpp8s* row1, const Rpp8s* row2,
                                     const Rpp8s* row3, const Rpp8s* row4, Rpp8s* dstRow,
                                     int width) {
    int j = 0;
    const int nlanes = 32;

    // Scalar left edge
    for (j = 0; j < std::min(2, width); j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8s)rpp_median_5x5_sortnet(p);
    }

    // Vector path
    for (; j < width - 2; j += nlanes) {
        if (j > width - 2 - nlanes) {
            if (j == 2 || (const Rpp8s*)dstRow == row2) break;
            j = width - 2 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - 2));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j - 2));
        __m256i p10 = _mm256_loadu_si256((const __m256i*)(row2 + j - 2));
        __m256i p15 = _mm256_loadu_si256((const __m256i*)(row3 + j - 2));
        __m256i p20 = _mm256_loadu_si256((const __m256i*)(row4 + j - 2));
        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j - 1));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row1 + j - 1));
        __m256i p11 = _mm256_loadu_si256((const __m256i*)(row2 + j - 1));
        __m256i p16 = _mm256_loadu_si256((const __m256i*)(row3 + j - 1));
        __m256i p21 = _mm256_loadu_si256((const __m256i*)(row4 + j - 1));
        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p12 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p17 = _mm256_loadu_si256((const __m256i*)(row3 + j));
        __m256i p22 = _mm256_loadu_si256((const __m256i*)(row4 + j));
        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row0 + j + 1));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row1 + j + 1));
        __m256i p13 = _mm256_loadu_si256((const __m256i*)(row2 + j + 1));
        __m256i p18 = _mm256_loadu_si256((const __m256i*)(row3 + j + 1));
        __m256i p23 = _mm256_loadu_si256((const __m256i*)(row4 + j + 1));
        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row0 + j + 2));
        __m256i p9 = _mm256_loadu_si256((const __m256i*)(row1 + j + 2));
        __m256i p14 = _mm256_loadu_si256((const __m256i*)(row2 + j + 2));
        __m256i p19 = _mm256_loadu_si256((const __m256i*)(row3 + j + 2));
        __m256i p24 = _mm256_loadu_si256((const __m256i*)(row4 + j + 2));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epi8(a, b); \
        b = _mm256_max_epi8(b, t); \
    }
        OP(p1, p2);
        OP(p0, p1);
        OP(p1, p2);
        OP(p4, p5);
        OP(p3, p4);
        OP(p4, p5);
        OP(p0, p3);
        OP(p2, p5);
        OP(p2, p3);
        OP(p1, p4);
        OP(p1, p2);
        OP(p3, p4);
        OP(p7, p8);
        OP(p6, p7);
        OP(p7, p8);
        OP(p10, p11);
        OP(p9, p10);
        OP(p10, p11);
        OP(p6, p9);
        OP(p8, p11);
        OP(p8, p9);
        OP(p7, p10);
        OP(p7, p8);
        OP(p9, p10);
        OP(p0, p6);
        OP(p4, p10);
        OP(p4, p6);
        OP(p2, p8);
        OP(p2, p4);
        OP(p6, p8);
        OP(p1, p7);
        OP(p5, p11);
        OP(p5, p7);
        OP(p3, p9);
        OP(p3, p5);
        OP(p7, p9);
        OP(p1, p2);
        OP(p3, p4);
        OP(p5, p6);
        OP(p7, p8);
        OP(p9, p10);
        OP(p13, p14);
        OP(p12, p13);
        OP(p13, p14);
        OP(p16, p17);
        OP(p15, p16);
        OP(p16, p17);
        OP(p12, p15);
        OP(p14, p17);
        OP(p14, p15);
        OP(p13, p16);
        OP(p13, p14);
        OP(p15, p16);
        OP(p19, p20);
        OP(p18, p19);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p21, p23);
        OP(p22, p24);
        OP(p22, p23);
        OP(p18, p21);
        OP(p20, p23);
        OP(p20, p21);
        OP(p19, p22);
        OP(p22, p24);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p12, p18);
        OP(p16, p22);
        OP(p16, p18);
        OP(p14, p20);
        OP(p20, p24);
        OP(p14, p16);
        OP(p18, p20);
        OP(p22, p24);
        OP(p13, p19);
        OP(p17, p23);
        OP(p17, p19);
        OP(p15, p21);
        OP(p15, p17);
        OP(p19, p21);
        OP(p13, p14);
        OP(p15, p16);
        OP(p17, p18);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p0, p12);
        OP(p8, p20);
        OP(p8, p12);
        OP(p4, p16);
        OP(p16, p24);
        OP(p12, p16);
        OP(p2, p14);
        OP(p10, p22);
        OP(p10, p14);
        OP(p6, p18);
        OP(p6, p10);
        OP(p10, p12);
        OP(p1, p13);
        OP(p9, p21);
        OP(p9, p13);
        OP(p5, p17);
        OP(p13, p17);
        OP(p3, p15);
        OP(p11, p23);
        OP(p11, p15);
        OP(p7, p19);
        OP(p7, p11);
        OP(p11, p13);
        OP(p11, p12);
#undef OP
        _mm256_storeu_si256((__m256i*)(dstRow + j), p12);
    }

    // Scalar tail
    for (; j < width; j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8s)rpp_median_5x5_sortnet(p);
    }
}

// 5×5 Median - F32 PLN1 (processes 8 pixels per iteration)
inline void rpp_median5x5_pln_f32_avx(const Rpp32f* row0, const Rpp32f* row1, const Rpp32f* row2,
                                      const Rpp32f* row3, const Rpp32f* row4, Rpp32f* dstRow,
                                      int width) {
    int j = 0;
    const int nlanes = 8;

    // Scalar left edge
    for (j = 0; j < std::min(2, width); j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        float p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                       row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                       row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                       row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = rpp_median_5x5_sortnet(p);
    }

    // Vector path
    for (; j < width - 2; j += nlanes) {
        if (j > width - 2 - nlanes) {
            if (j == 2 || (const Rpp32f*)dstRow == row2) break;
            j = width - 2 - nlanes;
        }

        __m256 p0 = _mm256_loadu_ps(row0 + j - 2);
        __m256 p5 = _mm256_loadu_ps(row1 + j - 2);
        __m256 p10 = _mm256_loadu_ps(row2 + j - 2);
        __m256 p15 = _mm256_loadu_ps(row3 + j - 2);
        __m256 p20 = _mm256_loadu_ps(row4 + j - 2);
        __m256 p1 = _mm256_loadu_ps(row0 + j - 1);
        __m256 p6 = _mm256_loadu_ps(row1 + j - 1);
        __m256 p11 = _mm256_loadu_ps(row2 + j - 1);
        __m256 p16 = _mm256_loadu_ps(row3 + j - 1);
        __m256 p21 = _mm256_loadu_ps(row4 + j - 1);
        __m256 p2 = _mm256_loadu_ps(row0 + j);
        __m256 p7 = _mm256_loadu_ps(row1 + j);
        __m256 p12 = _mm256_loadu_ps(row2 + j);
        __m256 p17 = _mm256_loadu_ps(row3 + j);
        __m256 p22 = _mm256_loadu_ps(row4 + j);
        __m256 p3 = _mm256_loadu_ps(row0 + j + 1);
        __m256 p8 = _mm256_loadu_ps(row1 + j + 1);
        __m256 p13 = _mm256_loadu_ps(row2 + j + 1);
        __m256 p18 = _mm256_loadu_ps(row3 + j + 1);
        __m256 p23 = _mm256_loadu_ps(row4 + j + 1);
        __m256 p4 = _mm256_loadu_ps(row0 + j + 2);
        __m256 p9 = _mm256_loadu_ps(row1 + j + 2);
        __m256 p14 = _mm256_loadu_ps(row2 + j + 2);
        __m256 p19 = _mm256_loadu_ps(row3 + j + 2);
        __m256 p24 = _mm256_loadu_ps(row4 + j + 2);

#define OP(a, b)                 \
    {                            \
        __m256 t = a;            \
        a = _mm256_min_ps(a, b); \
        b = _mm256_max_ps(b, t); \
    }
        OP(p1, p2);
        OP(p0, p1);
        OP(p1, p2);
        OP(p4, p5);
        OP(p3, p4);
        OP(p4, p5);
        OP(p0, p3);
        OP(p2, p5);
        OP(p2, p3);
        OP(p1, p4);
        OP(p1, p2);
        OP(p3, p4);
        OP(p7, p8);
        OP(p6, p7);
        OP(p7, p8);
        OP(p10, p11);
        OP(p9, p10);
        OP(p10, p11);
        OP(p6, p9);
        OP(p8, p11);
        OP(p8, p9);
        OP(p7, p10);
        OP(p7, p8);
        OP(p9, p10);
        OP(p0, p6);
        OP(p4, p10);
        OP(p4, p6);
        OP(p2, p8);
        OP(p2, p4);
        OP(p6, p8);
        OP(p1, p7);
        OP(p5, p11);
        OP(p5, p7);
        OP(p3, p9);
        OP(p3, p5);
        OP(p7, p9);
        OP(p1, p2);
        OP(p3, p4);
        OP(p5, p6);
        OP(p7, p8);
        OP(p9, p10);
        OP(p13, p14);
        OP(p12, p13);
        OP(p13, p14);
        OP(p16, p17);
        OP(p15, p16);
        OP(p16, p17);
        OP(p12, p15);
        OP(p14, p17);
        OP(p14, p15);
        OP(p13, p16);
        OP(p13, p14);
        OP(p15, p16);
        OP(p19, p20);
        OP(p18, p19);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p21, p23);
        OP(p22, p24);
        OP(p22, p23);
        OP(p18, p21);
        OP(p20, p23);
        OP(p20, p21);
        OP(p19, p22);
        OP(p22, p24);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p12, p18);
        OP(p16, p22);
        OP(p16, p18);
        OP(p14, p20);
        OP(p20, p24);
        OP(p14, p16);
        OP(p18, p20);
        OP(p22, p24);
        OP(p13, p19);
        OP(p17, p23);
        OP(p17, p19);
        OP(p15, p21);
        OP(p15, p17);
        OP(p19, p21);
        OP(p13, p14);
        OP(p15, p16);
        OP(p17, p18);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p0, p12);
        OP(p8, p20);
        OP(p8, p12);
        OP(p4, p16);
        OP(p16, p24);
        OP(p12, p16);
        OP(p2, p14);
        OP(p10, p22);
        OP(p10, p14);
        OP(p6, p18);
        OP(p6, p10);
        OP(p10, p12);
        OP(p1, p13);
        OP(p9, p21);
        OP(p9, p13);
        OP(p5, p17);
        OP(p13, p17);
        OP(p3, p15);
        OP(p11, p23);
        OP(p11, p15);
        OP(p7, p19);
        OP(p7, p11);
        OP(p11, p13);
        OP(p11, p12);
#undef OP
        _mm256_storeu_ps(dstRow + j, p12);
    }

    // Scalar tail
    for (; j < width; j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        float p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                       row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                       row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                       row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = rpp_median_5x5_sortnet(p);
    }
}

// 5×5 Median - F16 PLN1 (converts to F32, processes 8 pixels, converts back)
inline void rpp_median5x5_pln_f16_avx(const Rpp16f* row0, const Rpp16f* row1, const Rpp16f* row2,
                                      const Rpp16f* row3, const Rpp16f* row4, Rpp16f* dstRow,
                                      int width) {
    int j = 0;
    const int nlanes = 8;

    // Scalar left edge
    for (j = 0; j < std::min(2, width); j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        float p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                       row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                       row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                       row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = rpp_median_5x5_sortnet(p);
    }

    // Vector path
    for (; j < width - 2; j += nlanes) {
        if (j > width - 2 - nlanes) {
            if (j == 2 || (const Rpp16f*)dstRow == row2) break;
            j = width - 2 - nlanes;
        }

        __m256 p0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j - 2)));
        __m256 p5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j - 2)));
        __m256 p10 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j - 2)));
        __m256 p15 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row3 + j - 2)));
        __m256 p20 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row4 + j - 2)));
        __m256 p1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j - 1)));
        __m256 p6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j - 1)));
        __m256 p11 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j - 1)));
        __m256 p16 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row3 + j - 1)));
        __m256 p21 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row4 + j - 1)));
        __m256 p2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j)));
        __m256 p7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j)));
        __m256 p12 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j)));
        __m256 p17 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row3 + j)));
        __m256 p22 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row4 + j)));
        __m256 p3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j + 1)));
        __m256 p8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j + 1)));
        __m256 p13 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j + 1)));
        __m256 p18 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row3 + j + 1)));
        __m256 p23 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row4 + j + 1)));
        __m256 p4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + j + 2)));
        __m256 p9 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + j + 2)));
        __m256 p14 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row2 + j + 2)));
        __m256 p19 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row3 + j + 2)));
        __m256 p24 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row4 + j + 2)));

#define OP(a, b)                 \
    {                            \
        __m256 t = a;            \
        a = _mm256_min_ps(a, b); \
        b = _mm256_max_ps(b, t); \
    }
        OP(p1, p2);
        OP(p0, p1);
        OP(p1, p2);
        OP(p4, p5);
        OP(p3, p4);
        OP(p4, p5);
        OP(p0, p3);
        OP(p2, p5);
        OP(p2, p3);
        OP(p1, p4);
        OP(p1, p2);
        OP(p3, p4);
        OP(p7, p8);
        OP(p6, p7);
        OP(p7, p8);
        OP(p10, p11);
        OP(p9, p10);
        OP(p10, p11);
        OP(p6, p9);
        OP(p8, p11);
        OP(p8, p9);
        OP(p7, p10);
        OP(p7, p8);
        OP(p9, p10);
        OP(p0, p6);
        OP(p4, p10);
        OP(p4, p6);
        OP(p2, p8);
        OP(p2, p4);
        OP(p6, p8);
        OP(p1, p7);
        OP(p5, p11);
        OP(p5, p7);
        OP(p3, p9);
        OP(p3, p5);
        OP(p7, p9);
        OP(p1, p2);
        OP(p3, p4);
        OP(p5, p6);
        OP(p7, p8);
        OP(p9, p10);
        OP(p13, p14);
        OP(p12, p13);
        OP(p13, p14);
        OP(p16, p17);
        OP(p15, p16);
        OP(p16, p17);
        OP(p12, p15);
        OP(p14, p17);
        OP(p14, p15);
        OP(p13, p16);
        OP(p13, p14);
        OP(p15, p16);
        OP(p19, p20);
        OP(p18, p19);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p21, p23);
        OP(p22, p24);
        OP(p22, p23);
        OP(p18, p21);
        OP(p20, p23);
        OP(p20, p21);
        OP(p19, p22);
        OP(p22, p24);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p12, p18);
        OP(p16, p22);
        OP(p16, p18);
        OP(p14, p20);
        OP(p20, p24);
        OP(p14, p16);
        OP(p18, p20);
        OP(p22, p24);
        OP(p13, p19);
        OP(p17, p23);
        OP(p17, p19);
        OP(p15, p21);
        OP(p15, p17);
        OP(p19, p21);
        OP(p13, p14);
        OP(p15, p16);
        OP(p17, p18);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p0, p12);
        OP(p8, p20);
        OP(p8, p12);
        OP(p4, p16);
        OP(p16, p24);
        OP(p12, p16);
        OP(p2, p14);
        OP(p10, p22);
        OP(p10, p14);
        OP(p6, p18);
        OP(p6, p10);
        OP(p10, p12);
        OP(p1, p13);
        OP(p9, p21);
        OP(p9, p13);
        OP(p5, p17);
        OP(p13, p17);
        OP(p3, p15);
        OP(p11, p23);
        OP(p11, p15);
        OP(p7, p19);
        OP(p7, p11);
        OP(p11, p13);
        OP(p11, p12);
#undef OP
        _mm_storeu_si128((__m128i*)(dstRow + j), _mm256_cvtps_ph(p12, 0));
    }

    // Scalar tail
    for (; j < width; j++) {
        int j0 = std::max(j - 2, 0);
        int j1 = std::max(j - 1, 0);
        int j3 = std::min(j + 1, width - 1);
        int j4 = std::min(j + 2, width - 1);

        float p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                       row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                       row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                       row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = rpp_median_5x5_sortnet(p);
    }
}

// 3×3 Median - U8 PKD3 (processes 32 bytes per iteration)
inline void rpp_median3x3_pkd_u8_avx(const Rpp8u* row0, const Rpp8u* row1, const Rpp8u* row2,
                                     Rpp8u* dstRow, int widthBytes, int channels) {
    int j = 0;
    int limit = channels;

    // Scalar left edge
    for (; j < limit; j++) {
        int j0 = j >= channels ? j - channels : j;
        int j2 = j < widthBytes - channels ? j + channels : j;

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8u)rpp_median_3x3_sortnet(p);
    }

    const int nlanes = 32;
    for (; j < widthBytes - channels; j += nlanes) {
        if (j > widthBytes - channels - nlanes) {
            if (j == channels || (const Rpp8u*)dstRow == row1) break;
            j = widthBytes - channels - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - channels));
        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j + channels));
        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row1 + j - channels));
        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j + channels));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row2 + j - channels));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row2 + j + channels));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p1);
        OP(p3, p4);
        OP(p6, p7);
        OP(p1, p2);
        OP(p4, p5);
        OP(p7, p8);
        OP(p0, p3);
        OP(p5, p8);
        OP(p4, p7);
        OP(p3, p6);
        OP(p1, p4);
        OP(p2, p5);
        OP(p4, p7);
        OP(p4, p2);
        OP(p6, p4);
        OP(p4, p2);

#undef OP

        _mm256_storeu_si256((__m256i*)(dstRow + j), p4);
    }

    // Scalar tail/right edge
    for (; j < widthBytes; j++) {
        int j0 = j >= channels ? j - channels : j;
        int j2 = j < widthBytes - channels ? j + channels : j;

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8u)rpp_median_3x3_sortnet(p);
    }
}

// 5×5 Median - U8 PKD3 (processes 32 bytes per iteration)
inline void rpp_median5x5_pkd_u8_avx(const Rpp8u* row0, const Rpp8u* row1, const Rpp8u* row2,
                                     const Rpp8u* row3, const Rpp8u* row4, Rpp8u* dstRow,
                                     int widthBytes, int channels) {
    int j = 0;
    int limit = channels * 2;

    // Scalar left edge
    for (; j < limit; j++) {
        int j1 = j >= channels ? j - channels : j;
        int j0 = j >= channels * 2 ? j - channels * 2 : j1;
        int j3 = j < widthBytes - channels ? j + channels : j;
        int j4 = j < widthBytes - channels * 2 ? j + channels * 2 : j3;

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }

    const int nlanes = 32;
    for (; j < widthBytes - channels * 2; j += nlanes) {
        if (j > widthBytes - channels * 2 - nlanes) {
            if (j == channels * 2 || (const Rpp8u*)dstRow == row2) break;
            j = widthBytes - channels * 2 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(row0 + j - channels * 2));
        __m256i p5 = _mm256_loadu_si256((const __m256i*)(row1 + j - channels * 2));
        __m256i p10 = _mm256_loadu_si256((const __m256i*)(row2 + j - channels * 2));
        __m256i p15 = _mm256_loadu_si256((const __m256i*)(row3 + j - channels * 2));
        __m256i p20 = _mm256_loadu_si256((const __m256i*)(row4 + j - channels * 2));

        __m256i p1 = _mm256_loadu_si256((const __m256i*)(row0 + j - channels));
        __m256i p6 = _mm256_loadu_si256((const __m256i*)(row1 + j - channels));
        __m256i p11 = _mm256_loadu_si256((const __m256i*)(row2 + j - channels));
        __m256i p16 = _mm256_loadu_si256((const __m256i*)(row3 + j - channels));
        __m256i p21 = _mm256_loadu_si256((const __m256i*)(row4 + j - channels));

        __m256i p2 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i p7 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i p12 = _mm256_loadu_si256((const __m256i*)(row2 + j));
        __m256i p17 = _mm256_loadu_si256((const __m256i*)(row3 + j));
        __m256i p22 = _mm256_loadu_si256((const __m256i*)(row4 + j));

        __m256i p3 = _mm256_loadu_si256((const __m256i*)(row0 + j + channels));
        __m256i p8 = _mm256_loadu_si256((const __m256i*)(row1 + j + channels));
        __m256i p13 = _mm256_loadu_si256((const __m256i*)(row2 + j + channels));
        __m256i p18 = _mm256_loadu_si256((const __m256i*)(row3 + j + channels));
        __m256i p23 = _mm256_loadu_si256((const __m256i*)(row4 + j + channels));

        __m256i p4 = _mm256_loadu_si256((const __m256i*)(row0 + j + channels * 2));
        __m256i p9 = _mm256_loadu_si256((const __m256i*)(row1 + j + channels * 2));
        __m256i p14 = _mm256_loadu_si256((const __m256i*)(row2 + j + channels * 2));
        __m256i p19 = _mm256_loadu_si256((const __m256i*)(row3 + j + channels * 2));
        __m256i p24 = _mm256_loadu_si256((const __m256i*)(row4 + j + channels * 2));

#define OP(a, b)                   \
    {                              \
        __m256i t = a;             \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        OP(p1, p2);
        OP(p0, p1);
        OP(p1, p2);
        OP(p4, p5);
        OP(p3, p4);
        OP(p4, p5);
        OP(p0, p3);
        OP(p2, p5);
        OP(p2, p3);
        OP(p1, p4);
        OP(p1, p2);
        OP(p3, p4);
        OP(p7, p8);
        OP(p6, p7);
        OP(p7, p8);
        OP(p10, p11);
        OP(p9, p10);
        OP(p10, p11);
        OP(p6, p9);
        OP(p8, p11);
        OP(p8, p9);
        OP(p7, p10);
        OP(p7, p8);
        OP(p9, p10);
        OP(p0, p6);
        OP(p4, p10);
        OP(p4, p6);
        OP(p2, p8);
        OP(p2, p4);
        OP(p6, p8);
        OP(p1, p7);
        OP(p5, p11);
        OP(p5, p7);
        OP(p3, p9);
        OP(p3, p5);
        OP(p7, p9);
        OP(p1, p2);
        OP(p3, p4);
        OP(p5, p6);
        OP(p7, p8);
        OP(p9, p10);
        OP(p13, p14);
        OP(p12, p13);
        OP(p13, p14);
        OP(p16, p17);
        OP(p15, p16);
        OP(p16, p17);
        OP(p12, p15);
        OP(p14, p17);
        OP(p14, p15);
        OP(p13, p16);
        OP(p13, p14);
        OP(p15, p16);
        OP(p19, p20);
        OP(p18, p19);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p21, p23);
        OP(p22, p24);
        OP(p22, p23);
        OP(p18, p21);
        OP(p20, p23);
        OP(p20, p21);
        OP(p19, p22);
        OP(p22, p24);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p12, p18);
        OP(p16, p22);
        OP(p16, p18);
        OP(p14, p20);
        OP(p20, p24);
        OP(p14, p16);
        OP(p18, p20);
        OP(p22, p24);
        OP(p13, p19);
        OP(p17, p23);
        OP(p17, p19);
        OP(p15, p21);
        OP(p15, p17);
        OP(p19, p21);
        OP(p13, p14);
        OP(p15, p16);
        OP(p17, p18);
        OP(p19, p20);
        OP(p21, p22);
        OP(p23, p24);
        OP(p0, p12);
        OP(p8, p20);
        OP(p8, p12);
        OP(p4, p16);
        OP(p16, p24);
        OP(p12, p16);
        OP(p2, p14);
        OP(p10, p22);
        OP(p10, p14);
        OP(p6, p18);
        OP(p6, p10);
        OP(p10, p12);
        OP(p1, p13);
        OP(p9, p21);
        OP(p9, p13);
        OP(p5, p17);
        OP(p13, p17);
        OP(p3, p15);
        OP(p11, p23);
        OP(p11, p15);
        OP(p7, p19);
        OP(p7, p11);
        OP(p11, p13);
        OP(p11, p12);

#undef OP

        _mm256_storeu_si256((__m256i*)(dstRow + j), p12);
    }

    // Scalar tail/right edge
    for (; j < widthBytes; j++) {
        int j1 = j >= channels ? j - channels : j;
        int j0 = j >= channels * 2 ? j - channels * 2 : j1;
        int j3 = j < widthBytes - channels ? j + channels : j;
        int j4 = j < widthBytes - channels * 2 ? j + channels * 2 : j3;

        int p[25] = {row0[j0], row0[j1], row0[j],  row0[j3], row0[j4], row1[j0], row1[j1],
                     row1[j],  row1[j3], row1[j4], row2[j0], row2[j1], row2[j],  row2[j3],
                     row2[j4], row3[j0], row3[j1], row3[j],  row3[j3], row3[j4], row4[j0],
                     row4[j1], row4[j],  row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }
}

#endif  // __AVX2__

// -------------------- Scalar Sorting Network Fallback Functions --------------------
// Used when AVX2 is not available or for border pixels

template <typename T>
inline void median_filter_3x3_sortnet(T* srcPtrTemp, T* dstPtrTemp, Rpp32s rowIdx, Rpp32s colIdx,
                                      Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s channels,
                                      RpptDescPtr srcDescPtr) {
    using WT = std::conditional_t<std::is_integral<T>::value, int, T>;

    T blockData[9 * 4];  // channels <= 4
    Rpp32s index = 0;

    for (Rpp32s i = -1; i <= 1; i++) {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -1; j <= 1; j++) {
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
            else {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], channels * sizeof(T));
                index += channels;
            }
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++) {
        WT p[9];
        for (Rpp32s k = 0; k < 9; k++) p[k] = (WT)blockData[k * channels + ch];

        WT med = rpp_median_3x3_sortnet(p);
        dstPtrTemp[ch] = (T)med;
    }
}

template <typename T>
inline void median_filter_5x5_sortnet(T* srcPtrTemp, T* dstPtrTemp, Rpp32s rowIdx, Rpp32s colIdx,
                                      Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s channels,
                                      RpptDescPtr srcDescPtr) {
    using WT = std::conditional_t<std::is_integral<T>::value, int, T>;

    T blockData[25 * 4];  // channels <= 4
    Rpp32s index = 0;

    for (Rpp32s i = -2; i <= 2; i++) {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -2; j <= 2; j++) {
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
            else {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], channels * sizeof(T));
                index += channels;
            }
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++) {
        WT p[25];
        for (Rpp32s k = 0; k < 25; k++) p[k] = (WT)blockData[k * channels + ch];

        WT med = rpp_median_5x5_sortnet(p);
        dstPtrTemp[ch] = (T)med;
    }
}

// Generic median filter implementation
// Max kernel size is 9x9 = 81, max channels is 4, so max buffer size is 81 * 4 = 324
constexpr Rpp32s MAX_KERNEL_SIZE_SQUARED = 81;  // 9x9 kernel
constexpr Rpp32s MAX_CHANNELS = 4;

template <typename T>
inline void median_filter_generic(T* srcPtrTemp, T* dstPtrTemp, Rpp32s rowIdx, Rpp32s colIdx,
                                  Rpp32s kernelSizeSquared, Rpp32s padLength, Rpp32s heightLimit,
                                  Rpp32s widthLimit, Rpp32s channels, RpptDescPtr srcDescPtr,
                                  RpptDescPtr dstDescPtr) {
    // Fixed-size buffer to hold kernel window data for all channels
    // Using fixed-size array instead of VLA for C++ standard compliance and stack safety
    T blockData[MAX_KERNEL_SIZE_SQUARED * MAX_CHANNELS];
    Rpp32s index = 0, medianIndex = kernelSizeSquared / 2;

    // Fill blockData with padded values from the source image using nearest neighbor padding
    for (Rpp32s i = -padLength; i <= padLength; i++) {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -padLength; j <= padLength; j++) {
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            if (channels == 3) {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], 3 * sizeof(T));
                index += 3;
            } else if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++) {
        // Fixed-size buffer instead of VLA for C++ standard compliance
        T channelBlock[MAX_KERNEL_SIZE_SQUARED];

        for (Rpp32s i = 0; i < kernelSizeSquared; i++)
            channelBlock[i] = blockData[i * channels + ch];

        std::nth_element(channelBlock, channelBlock + medianIndex,
                         channelBlock + kernelSizeSquared);
        dstPtrTemp[ch] = channelBlock[medianIndex];
    }
}

template <typename T>
static inline RppStatus median_filter_generic_host_impl(T* srcPtrImage, RpptDescPtr srcDescPtr,
                                                        T* dstPtrImage, RpptDescPtr dstDescPtr,
                                                        Rpp32u kernelSize, RpptROI roi,
                                                        RppLayoutParams layoutParams) {
    T *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                    (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    Rpp32s kernelSizeSquared = kernelSize * kernelSize;
    Rpp32s padLength = kernelSize / 2;
    bool useSortNet3 = (kernelSize == 3);
    bool useSortNet5 = (kernelSize == 5);

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        // Use simplified histogram method for U8 PLN1 with large kernels
        if (std::is_same<T, Rpp8u>::value && kernelSize >= 7) {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++) {
                rpp_median_histogram_u8_pln1_host(
                    (const Rpp8u*)srcPtrChannel, (Rpp8u*)dstPtrChannel, roi.xywhROI.roiWidth,
                    roi.xywhROI.roiHeight, srcDescPtr->strides.hStride, dstDescPtr->strides.hStride,
                    kernelSize);
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        } else {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++) {
#if __AVX2__
                if ((useSortNet3 || useSortNet5) && std::is_same<T, Rpp8u>::value) {
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                        Rpp8u* dstRow = (Rpp8u*)dstPtrChannel + i * dstDescPtr->strides.hStride;
                        if (useSortNet3) {
                            const Rpp8u* r0 = (const Rpp8u*)srcPtrChannel +
                                              std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp8u* r1 =
                                (const Rpp8u*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp8u* r2 = (const Rpp8u*)srcPtrChannel +
                                              std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            rpp_median3x3_pln_u8_avx(r0, r1, r2, dstRow, roi.xywhROI.roiWidth);
                        } else {
                            const Rpp8u* r0 = (const Rpp8u*)srcPtrChannel +
                                              std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                            const Rpp8u* r1 = (const Rpp8u*)srcPtrChannel +
                                              std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp8u* r2 =
                                (const Rpp8u*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp8u* r3 = (const Rpp8u*)srcPtrChannel +
                                              std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            const Rpp8u* r4 = (const Rpp8u*)srcPtrChannel +
                                              std::min(i + 2, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            rpp_median5x5_pln_u8_avx(r0, r1, r2, r3, r4, dstRow,
                                                     roi.xywhROI.roiWidth);
                        }
                    }
                } else if ((useSortNet3 || useSortNet5) && std::is_same<T, Rpp8s>::value) {
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                        Rpp8s* dstRow = (Rpp8s*)dstPtrChannel + i * dstDescPtr->strides.hStride;
                        if (useSortNet3) {
                            const Rpp8s* r0 = (const Rpp8s*)srcPtrChannel +
                                              std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp8s* r1 =
                                (const Rpp8s*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp8s* r2 = (const Rpp8s*)srcPtrChannel +
                                              std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            rpp_median3x3_pln_i8_avx(r0, r1, r2, dstRow, roi.xywhROI.roiWidth);
                        } else {
                            const Rpp8s* r0 = (const Rpp8s*)srcPtrChannel +
                                              std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                            const Rpp8s* r1 = (const Rpp8s*)srcPtrChannel +
                                              std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp8s* r2 =
                                (const Rpp8s*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp8s* r3 = (const Rpp8s*)srcPtrChannel +
                                              std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            const Rpp8s* r4 = (const Rpp8s*)srcPtrChannel +
                                              std::min(i + 2, roi.xywhROI.roiHeight - 1) *
                                                  srcDescPtr->strides.hStride;
                            rpp_median5x5_pln_i8_avx(r0, r1, r2, r3, r4, dstRow,
                                                     roi.xywhROI.roiWidth);
                        }
                    }
                } else if ((useSortNet3 || useSortNet5) && std::is_same<T, Rpp32f>::value) {
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                        Rpp32f* dstRow = (Rpp32f*)dstPtrChannel + i * dstDescPtr->strides.hStride;
                        if (useSortNet3) {
                            const Rpp32f* r0 = (const Rpp32f*)srcPtrChannel +
                                               std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp32f* r1 =
                                (const Rpp32f*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp32f* r2 = (const Rpp32f*)srcPtrChannel +
                                               std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            rpp_median3x3_pln_f32_avx(r0, r1, r2, dstRow, roi.xywhROI.roiWidth);
                        } else {
                            const Rpp32f* r0 = (const Rpp32f*)srcPtrChannel +
                                               std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                            const Rpp32f* r1 = (const Rpp32f*)srcPtrChannel +
                                               std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp32f* r2 =
                                (const Rpp32f*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp32f* r3 = (const Rpp32f*)srcPtrChannel +
                                               std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            const Rpp32f* r4 = (const Rpp32f*)srcPtrChannel +
                                               std::min(i + 2, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            rpp_median5x5_pln_f32_avx(r0, r1, r2, r3, r4, dstRow,
                                                      roi.xywhROI.roiWidth);
                        }
                    }
                } else if ((useSortNet3 || useSortNet5) && std::is_same<T, Rpp16f>::value) {
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                        Rpp16f* dstRow = (Rpp16f*)dstPtrChannel + i * dstDescPtr->strides.hStride;
                        if (useSortNet3) {
                            const Rpp16f* r0 = (const Rpp16f*)srcPtrChannel +
                                               std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp16f* r1 =
                                (const Rpp16f*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp16f* r2 = (const Rpp16f*)srcPtrChannel +
                                               std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            rpp_median3x3_pln_f16_avx(r0, r1, r2, dstRow, roi.xywhROI.roiWidth);
                        } else {
                            const Rpp16f* r0 = (const Rpp16f*)srcPtrChannel +
                                               std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                            const Rpp16f* r1 = (const Rpp16f*)srcPtrChannel +
                                               std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                            const Rpp16f* r2 =
                                (const Rpp16f*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                            const Rpp16f* r3 = (const Rpp16f*)srcPtrChannel +
                                               std::min(i + 1, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            const Rpp16f* r4 = (const Rpp16f*)srcPtrChannel +
                                               std::min(i + 2, roi.xywhROI.roiHeight - 1) *
                                                   srcDescPtr->strides.hStride;
                            rpp_median5x5_pln_f16_avx(r0, r1, r2, r3, r4, dstRow,
                                                      roi.xywhROI.roiWidth);
                        }
                    }
                } else
#endif
                {
                    // Scalar fallback for non-AVX2 or unsupported types/kernel sizes
                    T* dstPtrRow = dstPtrChannel;
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                        T* dstPtrTemp = dstPtrRow;
                        for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++) {
                            if (useSortNet3)
                                median_filter_3x3_sortnet(srcPtrChannel, dstPtrTemp, i, j,
                                                          roi.xywhROI.roiHeight - 1,
                                                          roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                            else if (useSortNet5)
                                median_filter_5x5_sortnet(srcPtrChannel, dstPtrTemp, i, j,
                                                          roi.xywhROI.roiHeight - 1,
                                                          roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                            else
                                median_filter_generic(
                                    srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength,
                                    roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1,
                                    srcDescPtr, dstDescPtr);
                            dstPtrTemp++;
                        }
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
#if __AVX2__
        // Use simplified histogram method for U8 PKD with large kernels
        if (std::is_same<T, Rpp8u>::value && kernelSize >= 7) {
            rpp_median_histogram_u8_pkd_host(
                (const Rpp8u*)srcPtrChannel, (Rpp8u*)dstPtrChannel, roi.xywhROI.roiWidth,
                roi.xywhROI.roiHeight, srcDescPtr->strides.hStride, dstDescPtr->strides.hStride,
                kernelSize, srcDescPtr->c);
        }
        // AVX2 path for U8 PKD3 with 3x3 and 5x5 kernels
        else if (std::is_same<T, Rpp8u>::value && srcDescPtr->c == 3 &&
                 (useSortNet3 || useSortNet5)) {
            const int channels = 3;
            const int widthBytes = roi.xywhROI.roiWidth * channels;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u* dstRow = (Rpp8u*)dstPtrChannel + i * dstDescPtr->strides.hStride;

                if (useSortNet3) {
                    const Rpp8u* r0 = (const Rpp8u*)srcPtrChannel +
                                      std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                    const Rpp8u* r1 = (const Rpp8u*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                    const Rpp8u* r2 =
                        (const Rpp8u*)srcPtrChannel +
                        std::min(i + 1, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                    rpp_median3x3_pkd_u8_avx(r0, r1, r2, dstRow, widthBytes, channels);
                } else {
                    const Rpp8u* r0 = (const Rpp8u*)srcPtrChannel +
                                      std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                    const Rpp8u* r1 = (const Rpp8u*)srcPtrChannel +
                                      std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                    const Rpp8u* r2 = (const Rpp8u*)srcPtrChannel + i * srcDescPtr->strides.hStride;
                    const Rpp8u* r3 =
                        (const Rpp8u*)srcPtrChannel +
                        std::min(i + 1, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                    const Rpp8u* r4 =
                        (const Rpp8u*)srcPtrChannel +
                        std::min(i + 2, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                    rpp_median5x5_pkd_u8_avx(r0, r1, r2, r3, r4, dstRow, widthBytes, channels);
                }
            }
        } else
#endif
        {
            // Scalar fallback
            T* dstPtrRow = dstPtrChannel;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                T* dstPtrTemp = dstPtrRow;
                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++) {
                    if (useSortNet3)
                        median_filter_3x3_sortnet(
                            srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1,
                            roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr);
                    else if (useSortNet5)
                        median_filter_5x5_sortnet(
                            srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1,
                            roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr);
                    else
                        median_filter_generic(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared,
                                              padLength, roi.xywhROI.roiHeight - 1,
                                              roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr,
                                              dstDescPtr);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        T* dstPtrRow = dstPtrChannel;
        for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
            T* dstPtrTemp = dstPtrRow;
            for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++) {
                T* dstPtrTempChn = dstPtrTemp;
                T* srcPtrTempChn = srcPtrChannel;
                for (Rpp32s c = 0; c < srcDescPtr->c; c++) {
                    if (useSortNet3)
                        median_filter_3x3_sortnet(srcPtrTempChn, dstPtrTempChn, i, j,
                                                  roi.xywhROI.roiHeight - 1,
                                                  roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                    else if (useSortNet5)
                        median_filter_5x5_sortnet(srcPtrTempChn, dstPtrTempChn, i, j,
                                                  roi.xywhROI.roiHeight - 1,
                                                  roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                    else
                        median_filter_generic(srcPtrTempChn, dstPtrTempChn, i, j, kernelSizeSquared,
                                              padLength, roi.xywhROI.roiHeight - 1,
                                              roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                    srcPtrTempChn += srcDescPtr->strides.cStride;
                    dstPtrTempChn++;
                }
                dstPtrTemp += dstDescPtr->c;
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        for (Rpp32s c = 0; c < srcDescPtr->c; c++) {
            T* dstPtrRow = dstPtrChannel;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++) {
                T* dstPtrTemp = dstPtrRow;
                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++) {
                    if (useSortNet3)
                        median_filter_3x3_sortnet(srcPtrChannel, dstPtrTemp, i, j,
                                                  roi.xywhROI.roiHeight - 1,
                                                  roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                    else if (useSortNet5)
                        median_filter_5x5_sortnet(srcPtrChannel, dstPtrTemp, i, j,
                                                  roi.xywhROI.roiHeight - 1,
                                                  roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                    else
                        median_filter_generic(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared,
                                              padLength, roi.xywhROI.roiHeight - 1,
                                              roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus median_filter_generic_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                            RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                            RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                            RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        median_filter_generic_host_impl(srcPtrImage, srcDescPtr, dstPtrImage, dstDescPtr,
                                        kernelSize, roi, layoutParams);
    }

    return RPP_SUCCESS;
}

// -------------------- Single Image Processing --------------------

template <typename T>
RppStatus median_filter_generic_host_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                  RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                                  RpptROIPtr roiPtrSrc, RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    RpptROI roi;
    compute_roi_validation_host(roiPtrSrc, &roi, &roiDefault, roiType);
    return median_filter_generic_host_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr, kernelSize, roi,
                                           layoutParams);
}

template RppStatus median_filter_generic_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                            RpptDescPtr, Rpp32u, RpptROIPtr,
                                                            RpptRoiType, RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                            RpptDescPtr, Rpp32u, RpptROIPtr,
                                                            RpptRoiType, RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                             RpptDescPtr, Rpp32u, RpptROIPtr,
                                                             RpptRoiType, RppLayoutParams,
                                                             rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*,
                                                             RpptDescPtr, Rpp32u, RpptROIPtr,
                                                             RpptRoiType, RppLayoutParams,
                                                             rpp::Handle&);

template RppStatus median_filter_generic_host_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                                  RpptDescPtr, Rpp32u, RpptROIPtr,
                                                                  RpptRoiType, RppLayoutParams,
                                                                  rpp::Handle&);

template RppStatus median_filter_generic_host_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                                  RpptDescPtr, Rpp32u, RpptROIPtr,
                                                                  RpptRoiType, RppLayoutParams,
                                                                  rpp::Handle&);

template RppStatus median_filter_generic_host_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                                   RpptDescPtr, Rpp32u, RpptROIPtr,
                                                                   RpptRoiType, RppLayoutParams,
                                                                   rpp::Handle&);

template RppStatus median_filter_generic_host_single_image<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*,
                                                                   RpptDescPtr, Rpp32u, RpptROIPtr,
                                                                   RpptRoiType, RppLayoutParams,
                                                                   rpp::Handle&);

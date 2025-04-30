/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

//
//
// THIS FILE CONTAINS VERY FEW ROUTINES FOR:
//   - using a Random Number Generator, we use the most basic one for extreme simplicity
//   - random initialization of dense vectors.
//   - initializing a sparse matrix corresponding to 9-points stencil 2D-Laplacian matrix with few sparse formats (csr, coo and ell).
//   - some utilities (get rocsparse_indextype/_datatype from standard types)
//  No more.
//
//

#include <rocsparse/rocsparse.h>
#include <stdlib.h>
#include <vector>

template <typename I>
inline rocsparse_indextype utils_indextype(void);
template <typename T>
inline rocsparse_datatype utils_datatype(void);

//
// Not intended for integral types (instances are only numeric in this file).
//
template <typename T>
inline T utils_random(T a = static_cast<T>(0), T b = static_cast<T>(1));

//
// @brief Convert csr indexing to coordinates indexing.
// @param M number of rows of the csr indexing.
// @param nnz number of non-zeros of the csr indexing.
// @param csr_row_ptr indices to the beginning of each row of the csr indexing.
// @param coo_row_ind indices to the beginning of each row of the csr indexing.
// @param base base index.
//
template <typename I, typename J>
inline void utils_csr_to_coo(J                     M,
                             I                     nnz,
                             const std::vector<I>& csr_row_ptr,
                             std::vector<J>&       coo_row_ind,
                             rocsparse_index_base  base);

template <typename I, typename J, typename T>
inline void utils_csr_to_ell(J                     M,
                             const std::vector<I>& csr_row_ptr,
                             const std::vector<J>& csr_col_ind,
                             const std::vector<T>& csr_val,
                             std::vector<J>&       ell_col_ind,
                             std::vector<T>&       ell_val,
                             J&                    ell_width,
                             rocsparse_index_base  csr_base,
                             rocsparse_index_base  ell_base);

template <typename T>
inline void utils_init(T*     A,
                       size_t M,
                       size_t N,
                       size_t lda,
                       size_t stride      = 0,
                       size_t batch_count = 1,
                       T      a           = static_cast<T>(0),
                       T      b           = static_cast<T>(1));

template <typename T>
inline void utils_init(std::vector<T>& A,
                       size_t          M,
                       size_t          N,
                       size_t          lda,
                       size_t          stride      = 0,
                       size_t          batch_count = 1,
                       T               a           = static_cast<T>(0),
                       T               b           = static_cast<T>(1));

template <typename I, typename J, typename T>
inline void utils_init_csr_laplace2d(std::vector<I>&      row_ptr,
                                     std::vector<J>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     J&                   M,
                                     J&                   N,
                                     I&                   nnz,
                                     rocsparse_index_base base);
template <typename I, typename T>
inline void utils_init_ell_laplace2d(std::vector<I>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     I&                   M,
                                     I&                   N,
                                     I&                   width,
                                     rocsparse_index_base base);

template <typename I, typename T>
inline void utils_init_coo_laplace2d(std::vector<I>&      row_ind,
                                     std::vector<I>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     I&                   M,
                                     I&                   N,
                                     I&                   nnz,
                                     rocsparse_index_base base);

template <>
inline double utils_random<double>(double a, double b)
{
    const double t = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    return a * (static_cast<double>(1) - t) + b * t;
}

template <>
inline float utils_random<float>(float a, float b)
{
    const float t = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return a * (static_cast<float>(1) - t) + b * t;
}

template <>
inline rocsparse_float_complex utils_random<rocsparse_float_complex>(rocsparse_float_complex a,
                                                                     rocsparse_float_complex b)
{
    float theta = utils_random<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = utils_random<float>(std::abs(a), std::abs(b));

    return rocsparse_float_complex(r * cos(theta), r * sin(theta));
}

template <>
inline rocsparse_double_complex utils_random<rocsparse_double_complex>(rocsparse_double_complex a,
                                                                       rocsparse_double_complex b)
{

    double theta = utils_random<double>(0.0, 2.0 * acos(-1.0));
    double r     = utils_random<double>(std::abs(a), std::abs(b));
    return rocsparse_double_complex(r * cos(theta), r * sin(theta));
}

inline void utils_seedrand()
{
    srand(0);
}

template <>
inline rocsparse_indextype utils_indextype<uint16_t>(void)
{
    return rocsparse_indextype_u16;
}

template <>
inline rocsparse_indextype utils_indextype<int32_t>(void)
{
    return rocsparse_indextype_i32;
}

template <>
inline rocsparse_indextype utils_indextype<int64_t>(void)
{
    return rocsparse_indextype_i64;
}

template <>
inline rocsparse_datatype utils_datatype<float>(void)
{
    return rocsparse_datatype_f32_r;
}

template <>
inline rocsparse_datatype utils_datatype<double>(void)
{
    return rocsparse_datatype_f64_r;
}

template <>
inline rocsparse_datatype utils_datatype<rocsparse_float_complex>(void)
{
    return rocsparse_datatype_f32_c;
}

template <>
inline rocsparse_datatype utils_datatype<rocsparse_double_complex>(void)
{
    return rocsparse_datatype_f64_c;
}

inline double utils_time_us(void)
{
    auto now = std::chrono::steady_clock::now();
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

template <typename T>
inline void
    utils_init(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, T a, T b)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t j = 0; j < N; ++j)
            for(size_t i = 0; i < M; ++i)
            {
                A[i + j * lda + i_batch * stride] = utils_random<T>(a, b);
            }
}

template <typename T>
inline void utils_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, T a, T b)
{
    utils_init(A.data(), M, N, lda, stride, batch_count, a, b);
}

template <typename I, typename J, typename T>
inline void utils_init_csr_laplace2d(std::vector<I>&      row_ptr,
                                     std::vector<J>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     J&                   M,
                                     J&                   N,
                                     I&                   nnz,
                                     rocsparse_index_base base)
{
    // Do nothing
    if(dim_x == 0 || dim_y == 0)
    {
        return;
    }

    M = dim_x * dim_y;
    N = dim_x * dim_y;

    // Approximate 9pt stencil
    I nnz_mat = 9 * M;

    row_ptr.resize(M + 1);
    col_ind.resize(nnz_mat);
    val.resize(nnz_mat);

    nnz        = base;
    row_ptr[0] = base;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iy = 0; iy < dim_y; ++iy)
    {
        for(int32_t ix = 0; ix < dim_x; ++ix)
        {
            J row = iy * dim_x + ix;

            for(int32_t sy = -1; sy <= 1; ++sy)
            {
                if(iy + sy > -1 && iy + sy < dim_y)
                {
                    for(int32_t sx = -1; sx <= 1; ++sx)
                    {
                        if(ix + sx > -1 && ix + sx < dim_x)
                        {
                            J col = row + sy * dim_x + sx;

                            col_ind[nnz - base] = col + base;
                            val[nnz - base]     = (col == row) ? 8.0 : -1.0;

                            ++nnz;
                        }
                    }
                }
            }

            row_ptr[row + 1] = nnz;
        }
    }

    // Adjust nnz by index base
    nnz -= base;
}

template <typename I, typename T>
inline void utils_init_ell_laplace2d(std::vector<I>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     I&                   M,
                                     I&                   N,
                                     I&                   width,
                                     rocsparse_index_base base)
{
    I csr_nnz;

    std::vector<I> csr_row_ptr;
    std::vector<I> csr_col_ind;
    std::vector<T> csr_val;

    // Sample CSR matrix
    utils_init_csr_laplace2d(csr_row_ptr, csr_col_ind, csr_val, dim_x, dim_y, M, N, csr_nnz, base);

    // Convert to ELL
    utils_csr_to_ell(M, csr_row_ptr, csr_col_ind, csr_val, col_ind, val, width, base, base);
}

template <typename I, typename T>
inline void utils_init_coo_laplace2d(std::vector<I>&      row_ind,
                                     std::vector<I>&      col_ind,
                                     std::vector<T>&      val,
                                     int32_t              dim_x,
                                     int32_t              dim_y,
                                     I&                   M,
                                     I&                   N,
                                     I&                   nnz,
                                     rocsparse_index_base base)
{
    std::vector<I> row_ptr;

    // Sample CSR matrix
    utils_init_csr_laplace2d(row_ptr, col_ind, val, dim_x, dim_y, M, N, nnz, base);

    // Convert to COO
    utils_csr_to_coo(M, nnz, row_ptr, row_ind, base);
}

template <typename I, typename J>
inline void utils_csr_to_coo(J                     M,
                             I                     nnz,
                             const std::vector<I>& csr_row_ptr,
                             std::vector<J>&       coo_row_ind,
                             rocsparse_index_base  base)
{
    // Resize coo_row_ind
    coo_row_ind.resize(nnz);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;
        for(I j = row_begin; j < row_end; ++j)
        {
            coo_row_ind[j] = i + base;
        }
    }
}

template <typename I, typename J, typename T>
inline void utils_csr_to_ell(J                     M,
                             const std::vector<I>& csr_row_ptr,
                             const std::vector<J>& csr_col_ind,
                             const std::vector<T>& csr_val,
                             std::vector<J>&       ell_col_ind,
                             std::vector<T>&       ell_val,
                             J&                    ell_width,
                             rocsparse_index_base  csr_base,
                             rocsparse_index_base  ell_base)
{
    // Determine ELL width
    ell_width = 0;

    for(J i = 0; i < M; ++i)
    {
        J row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        ell_width = std::max(row_nnz, ell_width);
    }

    // Compute ELL non-zeros
    I ell_nnz = ell_width * M;

    ell_col_ind.resize(ell_nnz);
    ell_val.resize(ell_nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        J p = 0;

        I row_begin = csr_row_ptr[i] - csr_base;
        I row_end   = csr_row_ptr[i + 1] - csr_base;
        J row_nnz   = row_end - row_begin;

        // Fill ELL matrix with data
        for(I j = row_begin; j < row_end; ++j)
        {
            I idx = p * M + i;

            ell_col_ind[idx] = csr_col_ind[j] - csr_base + ell_base;
            ell_val[idx]     = csr_val[j];

            ++p;
        }

        // Add padding to ELL structures
        for(J j = row_nnz; j < ell_width; ++j)
        {
            I idx = p * M + i;

            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);

            ++p;
        }
    }
}

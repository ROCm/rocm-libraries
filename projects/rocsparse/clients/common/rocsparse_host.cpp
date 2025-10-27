/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_clients_routine_trace.hpp"
#include "utility.hpp"

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) \
    ((dir == rocsparse_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename X, typename Y>
void host_axpby(
    I size, I nnz, T alpha, const X* x_val, const I* x_ind, T beta, Y* y, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    for(I i = 0; i < size; ++i)
    {
        y[i] *= beta;
    }

    for(I i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = std::fma(
            static_cast<T>(alpha), static_cast<T>(x_val[i]), static_cast<T>(y[x_ind[i] - base]));
    }
}

template <typename I, typename X, typename Y, typename T>
void host_doti(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(
            static_cast<T>(y[x_ind[i] - base]), static_cast<T>(x_val[i]), static_cast<T>(*result));
    }
}

template <typename I, typename X, typename Y, typename T>
void host_dotci(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    *result = static_cast<T>(0);

    for(I i = 0; i < nnz; ++i)
    {
        *result = std::fma(static_cast<T>(rocsparse_conj(x_val[i])),
                           static_cast<T>(y[x_ind[i] - base]),
                           static_cast<T>(*result));
    }
}

template <typename I, typename T>
void host_gthr(I nnz, const T* y, T* x_val, const I* x_ind, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    for(I i = 0; i < nnz; ++i)
    {
        x_val[i] = y[x_ind[i] - base];
    }
}

template <typename T>
void host_gthrz(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        x_val[i]           = y[x_ind[i] - base];
        y[x_ind[i] - base] = static_cast<T>(0);
    }
}

template <typename I, typename T>
void host_roti(
    I nnz, T* x_val, const I* x_ind, T* y, const T* c, const T* s, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    for(I i = 0; i < nnz; ++i)
    {
        I idx = x_ind[i] - base;

        T xs = x_val[i];
        T ys = y[idx];

        x_val[i] = std::fma(*c, xs, *s * ys);
        y[idx]   = std::fma(*c, ys, -*s * xs);
    }
}

template <typename I, typename T>
void host_sctr(I nnz, const T* x_val, const I* x_ind, T* y, rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    for(I i = 0; i < nnz; ++i)
    {
        y[x_ind[i] - base] = x_val[i];
    }
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_bsrmv(rocsparse_direction  dir,
                rocsparse_operation  trans,
                J                    mb,
                J                    nb,
                I                    nnzb,
                T                    alpha,
                const I*             bsr_row_ptr,
                const I*             bsr_end_ptr,
                const J*             bsr_col_ind,
                const A*             bsr_val,
                J                    bsr_dim,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Quick return
    if(alpha == static_cast<T>(0))
    {
        if(beta != static_cast<T>(1))
        {
            for(J i = 0; i < mb * bsr_dim; ++i)
            {
                y[i] *= beta;
            }
        }

        return;
    }

    uint32_t WFSIZE;

    if(bsr_dim == 2)
    {
        I blocks_per_row = (mb != 0) ? (nnzb / mb) : 0;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(bsr_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(bsr_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J row = 0; row < mb; ++row)
    {
        I row_begin = bsr_row_ptr[row] - base;
        I row_end   = bsr_end_ptr[row] - base;

        if(bsr_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));

            for(I j = row_begin; j < row_end; j += WFSIZE)
            {
                for(uint32_t k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        J col = bsr_col_ind[j + k] - base;

                        if(dir == rocsparse_direction_column)
                        {
                            sum0[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 0]),
                                           static_cast<T>(x[col * bsr_dim + 0]),
                                           static_cast<T>(sum0[k]));
                            sum1[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 1]),
                                           static_cast<T>(x[col * bsr_dim + 0]),
                                           static_cast<T>(sum1[k]));
                            sum0[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 2]),
                                           static_cast<T>(x[col * bsr_dim + 1]),
                                           static_cast<T>(sum0[k]));
                            sum1[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 3]),
                                           static_cast<T>(x[col * bsr_dim + 1]),
                                           static_cast<T>(sum1[k]));
                        }
                        else
                        {
                            sum0[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 0]),
                                           static_cast<T>(x[col * bsr_dim + 0]),
                                           static_cast<T>(sum0[k]));
                            sum0[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 1]),
                                           static_cast<T>(x[col * bsr_dim + 1]),
                                           static_cast<T>(sum0[k]));
                            sum1[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 2]),
                                           static_cast<T>(x[col * bsr_dim + 0]),
                                           static_cast<T>(sum1[k]));
                            sum1[k]
                                = std::fma(static_cast<T>(bsr_val[bsr_dim * bsr_dim * (j + k) + 3]),
                                           static_cast<T>(x[col * bsr_dim + 1]),
                                           static_cast<T>(sum1[k]));
                        }
                    }
                }
            }

            for(uint32_t j = 1; j < WFSIZE; j <<= 1)
            {
                for(uint32_t k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * bsr_dim + 0] = std::fma(static_cast<T>(beta),
                                                static_cast<T>(y[row * bsr_dim + 0]),
                                                static_cast<T>(alpha * sum0[0]));
                y[row * bsr_dim + 1] = std::fma(static_cast<T>(beta),
                                                static_cast<T>(y[row * bsr_dim + 1]),
                                                static_cast<T>(alpha * sum1[0]));
            }
            else
            {
                y[row * bsr_dim + 0] = alpha * sum0[0];
                y[row * bsr_dim + 1] = alpha * sum1[0];
            }
        }
        else
        {
            for(J bi = 0; bi < bsr_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, static_cast<T>(0));

                for(I j = row_begin; j < row_end; ++j)
                {
                    J col = bsr_col_ind[j] - base;

                    for(J bj = 0; bj < bsr_dim; bj += WFSIZE)
                    {
                        for(uint32_t k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < bsr_dim)
                            {
                                if(dir == rocsparse_direction_column)
                                {
                                    sum[k] = std::fma(
                                        static_cast<T>(bsr_val[bsr_dim * bsr_dim * j
                                                               + bsr_dim * (bj + k) + bi]),
                                        static_cast<T>(x[bsr_dim * col + (bj + k)]),
                                        static_cast<T>(sum[k]));
                                }
                                else
                                {
                                    sum[k] = std::fma(
                                        static_cast<T>(bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi
                                                               + (bj + k)]),
                                        static_cast<T>(x[bsr_dim * col + (bj + k)]),
                                        static_cast<T>(sum[k]));
                                }
                            }
                        }
                    }
                }

                for(uint32_t j = 1; j < WFSIZE; j <<= 1)
                {
                    for(uint32_t k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta != static_cast<T>(0))
                {
                    y[row * bsr_dim + bi] = std::fma(static_cast<T>(beta),
                                                     static_cast<T>(y[row * bsr_dim + bi]),
                                                     static_cast<T>(alpha * sum[0]));
                }
                else
                {
                    y[row * bsr_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_bsrmv(rocsparse_direction  dir,
                rocsparse_operation  trans,
                J                    mb,
                J                    nb,
                I                    nnzb,
                T                    alpha,
                const I*             bsr_row_ptr,
                const J*             bsr_col_ind,
                const A*             bsr_val,
                J                    bsr_dim,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return host_bsrmv(dir,
                      trans,
                      mb,
                      nb,
                      nnzb,
                      alpha,
                      bsr_row_ptr,
                      bsr_row_ptr + 1,
                      bsr_col_ind,
                      bsr_val,
                      bsr_dim,
                      x,
                      beta,
                      y,
                      base);
}

template <typename T>
void host_bsrxmv(rocsparse_direction  dir,
                 rocsparse_operation  trans,
                 rocsparse_int        size_of_mask,
                 rocsparse_int        mb,
                 rocsparse_int        nb,
                 rocsparse_int        nnzb,
                 T                    alpha,
                 const rocsparse_int* bsr_mask_ptr,
                 const rocsparse_int* bsr_row_ptr,
                 const rocsparse_int* bsr_end_ptr,
                 const rocsparse_int* bsr_col_ind,
                 const T*             bsr_val,
                 rocsparse_int        bsr_dim,
                 const T*             x,
                 T                    beta,
                 T*                   y,
                 rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(bsr_mask_ptr == nullptr)
    {
        return host_bsrmv(dir,
                          trans,
                          mb,
                          nb,
                          nnzb,
                          alpha,
                          bsr_row_ptr,
                          bsr_end_ptr,
                          bsr_col_ind,
                          bsr_val,
                          bsr_dim,
                          x,
                          beta,
                          y,
                          base);
    }

    // Quick return
    if(alpha == static_cast<T>(0))
    {
        if(beta != static_cast<T>(1))
        {
            for(rocsparse_int i = 0; i < size_of_mask; ++i)
            {
                rocsparse_int shift = (bsr_mask_ptr[i] - base) * bsr_dim;
                for(rocsparse_int j = 0; j < bsr_dim; ++j)
                {
                    y[shift + j] *= beta;
                }
            }
        }

        return;
    }

    rocsparse_int WFSIZE;

    if(bsr_dim == 2)
    {
        rocsparse_int blocks_per_row = (mb != 0) ? (nnzb / mb) : 0;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(bsr_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(bsr_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int mask_idx = 0; mask_idx < size_of_mask; ++mask_idx)
    {
        rocsparse_int row       = bsr_mask_ptr[mask_idx] - base;
        rocsparse_int row_begin = bsr_row_ptr[row] - base;
        rocsparse_int row_end   = bsr_end_ptr[row] - base;

        if(bsr_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        if(dir == rocsparse_direction_column)
                        {
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                               x[col * bsr_dim + 0],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                               x[col * bsr_dim + 0],
                                               sum1[k]);
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                               x[col * bsr_dim + 1],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                               x[col * bsr_dim + 1],
                                               sum1[k]);
                        }
                        else
                        {
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                               x[col * bsr_dim + 0],
                                               sum0[k]);
                            sum0[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                               x[col * bsr_dim + 1],
                                               sum0[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                               x[col * bsr_dim + 0],
                                               sum1[k]);
                            sum1[k] = std::fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                               x[col * bsr_dim + 1],
                                               sum1[k]);
                        }
                    }
                }
            }

            for(uint32_t j = 1; j < WFSIZE; j <<= 1)
            {
                for(uint32_t k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * bsr_dim + 0] = std::fma(beta, y[row * bsr_dim + 0], alpha * sum0[0]);
                y[row * bsr_dim + 1] = std::fma(beta, y[row * bsr_dim + 1], alpha * sum1[0]);
            }
            else
            {
                y[row * bsr_dim + 0] = alpha * sum0[0];
                y[row * bsr_dim + 1] = alpha * sum1[0];
            }
        }
        else
        {
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, static_cast<T>(0));

                for(rocsparse_int j = row_begin; j < row_end; ++j)
                {
                    rocsparse_int col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = 0; bj < bsr_dim; bj += WFSIZE)
                    {
                        for(uint32_t k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < bsr_dim)
                            {
                                if(dir == rocsparse_direction_column)
                                {
                                    sum[k] = std::fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * (bj + k) + bi],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                                else
                                {
                                    sum[k] = std::fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi + (bj + k)],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                            }
                        }
                    }
                }

                for(uint32_t j = 1; j < WFSIZE; j <<= 1)
                {
                    for(uint32_t k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta != static_cast<T>(0))
                {
                    y[row * bsr_dim + bi] = std::fma(beta, y[row * bsr_dim + bi], alpha * sum[0]);
                }
                else
                {
                    y[row * bsr_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

template <typename T>
void host_gebsrmv(rocsparse_direction  dir,
                  rocsparse_operation  trans,
                  rocsparse_int        mb,
                  rocsparse_int        nb,
                  rocsparse_int        nnzb,
                  T                    alpha,
                  const rocsparse_int* bsr_row_ptr,
                  const rocsparse_int* bsr_col_ind,
                  const T*             bsr_val,
                  rocsparse_int        row_block_dim,
                  rocsparse_int        col_block_dim,
                  const T*             x,
                  T                    beta,
                  T*                   y,
                  rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Quick return
    if(alpha == static_cast<T>(0))
    {
        if(beta != static_cast<T>(1))
        {
            for(rocsparse_int i = 0; i < mb * row_block_dim; ++i)
            {
                y[i] *= beta;
            }
        }

        return;
    }

    if(row_block_dim == col_block_dim)
    {
        host_bsrmv(dir,
                   trans,
                   mb,
                   nb,
                   nnzb,
                   alpha,
                   bsr_row_ptr,
                   bsr_col_ind,
                   bsr_val,
                   row_block_dim,
                   x,
                   beta,
                   y,
                   base);

        return;
    }

    rocsparse_int WFSIZE;

    if(row_block_dim == 2 || row_block_dim == 3 || row_block_dim == 4)
    {
        rocsparse_int blocks_per_row = (mb != 0) ? (nnzb / mb) : 0;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(row_block_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(row_block_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int row = 0; row < mb; ++row)
    {
        rocsparse_int row_begin = bsr_row_ptr[row] - base;
        rocsparse_int row_end   = bsr_row_ptr[row + 1] - base;

        if(row_block_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                            }
                        }
                    }
                }
            }

            for(uint32_t j = 1; j < WFSIZE; j <<= 1)
            {
                for(uint32_t k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
            }
        }
        else if(row_block_dim == 3)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));
            std::vector<T> sum2(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 2],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 2 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                            }
                        }
                    }
                }
            }

            for(uint32_t j = 1; j < WFSIZE; j <<= 1)
            {
                for(uint32_t k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                    sum2[k] += sum2[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
                y[row * row_block_dim + 2]
                    = std::fma(beta, y[row * row_block_dim + 2], alpha * sum2[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
                y[row * row_block_dim + 2] = alpha * sum2[0];
            }
        }
        else if(row_block_dim == 4)
        {
            std::vector<T> sum0(WFSIZE, static_cast<T>(0));
            std::vector<T> sum1(WFSIZE, static_cast<T>(0));
            std::vector<T> sum2(WFSIZE, static_cast<T>(0));
            std::vector<T> sum3(WFSIZE, static_cast<T>(0));

            for(rocsparse_int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(rocsparse_int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        rocsparse_int col = bsr_col_ind[j + k] - base;

                        for(rocsparse_int l = 0; l < col_block_dim; l++)
                        {
                            if(dir == rocsparse_direction_column)
                            {
                                sum0[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l],
                                                   x[col * col_block_dim + l],
                                                   sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 1],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 2],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                                sum3[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + row_block_dim * l + 3],
                                                   x[col * col_block_dim + l],
                                                   sum3[k]);
                            }
                            else
                            {
                                sum0[k]
                                    = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k) + l],
                                               x[col * col_block_dim + l],
                                               sum0[k]);
                                sum1[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum1[k]);
                                sum2[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 2 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum2[k]);
                                sum3[k] = std::fma(bsr_val[row_block_dim * col_block_dim * (j + k)
                                                           + 3 * col_block_dim + l],
                                                   x[col * col_block_dim + l],
                                                   sum3[k]);
                            }
                        }
                    }
                }
            }

            for(uint32_t j = 1; j < WFSIZE; j <<= 1)
            {
                for(uint32_t k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] += sum0[k + j];
                    sum1[k] += sum1[k + j];
                    sum2[k] += sum2[k + j];
                    sum3[k] += sum3[k + j];
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[row * row_block_dim + 0]
                    = std::fma(beta, y[row * row_block_dim + 0], alpha * sum0[0]);
                y[row * row_block_dim + 1]
                    = std::fma(beta, y[row * row_block_dim + 1], alpha * sum1[0]);
                y[row * row_block_dim + 2]
                    = std::fma(beta, y[row * row_block_dim + 2], alpha * sum2[0]);
                y[row * row_block_dim + 3]
                    = std::fma(beta, y[row * row_block_dim + 3], alpha * sum3[0]);
            }
            else
            {
                y[row * row_block_dim + 0] = alpha * sum0[0];
                y[row * row_block_dim + 1] = alpha * sum1[0];
                y[row * row_block_dim + 2] = alpha * sum2[0];
                y[row * row_block_dim + 3] = alpha * sum3[0];
            }
        }
        else
        {
            for(rocsparse_int bi = 0; bi < row_block_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, static_cast<T>(0));

                for(rocsparse_int j = row_begin; j < row_end; ++j)
                {
                    rocsparse_int col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = 0; bj < col_block_dim; bj += WFSIZE)
                    {
                        for(uint32_t k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < col_block_dim)
                            {
                                if(dir == rocsparse_direction_column)
                                {
                                    sum[k] = std::fma(bsr_val[row_block_dim * col_block_dim * j
                                                              + row_block_dim * (bj + k) + bi],
                                                      x[col_block_dim * col + (bj + k)],
                                                      sum[k]);
                                }
                                else
                                {
                                    sum[k] = std::fma(bsr_val[row_block_dim * col_block_dim * j
                                                              + col_block_dim * bi + (bj + k)],
                                                      x[col_block_dim * col + (bj + k)],
                                                      sum[k]);
                                }
                            }
                        }
                    }
                }

                for(uint32_t j = 1; j < WFSIZE; j <<= 1)
                {
                    for(uint32_t k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta != static_cast<T>(0))
                {
                    y[row * row_block_dim + bi]
                        = std::fma(beta, y[row * row_block_dim + bi], alpha * sum[0]);
                }
                else
                {
                    y[row * row_block_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

template <typename T>
static inline void host_bsr_lsolve(rocsparse_direction  dir,
                                   rocsparse_operation  trans_X,
                                   rocsparse_int        mb,
                                   rocsparse_int        nrhs,
                                   T                    alpha,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   const T*             bsr_val,
                                   rocsparse_int        bsr_dim,
                                   const T*             B,
                                   int64_t              ldb,
                                   T*                   X,
                                   int64_t              ldx,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_index_base base,
                                   rocsparse_int*       struct_pivot,
                                   rocsparse_int*       numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(rocsparse_int i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(rocsparse_int bsr_row = 0; bsr_row < mb; ++bsr_row)
        {
            rocsparse_int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            rocsparse_int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            // Loop over blocks rows
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                rocsparse_int diag      = -1;
                rocsparse_int local_row = bsr_row * bsr_dim + bi;

                int64_t idx_B = (trans_X == rocsparse_operation_none) ? i * ldb + local_row
                                                                      : local_row * ldb + i;
                int64_t idx_X = (trans_X == rocsparse_operation_none) ? i * ldx + local_row
                                                                      : local_row * ldx + i;

                T sum      = alpha * B[idx_B];
                T diag_val = static_cast<T>(0);

                // Loop over BSR columns
                for(rocsparse_int j = bsr_row_begin; j < bsr_row_end; ++j)
                {
                    rocsparse_int bsr_col = bsr_col_ind[j] - base;

                    // Loop over blocks columns
                    for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                    {
                        rocsparse_int local_col = bsr_col * bsr_dim + bj;
                        T             local_val = (dir == rocsparse_direction_row)
                                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        if(local_val == static_cast<T>(0) && local_col == local_row
                           && diag_type == rocsparse_diag_type_non_unit)
                        {
                            // Numerical zero pivot found, avoid division by 0
                            // and store index for later use.
                            *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                            local_val      = static_cast<T>(1);
                        }

                        // Ignore all entries that are above the diagonal
                        if(local_col > local_row)
                        {
                            break;
                        }

                        // Diagonal
                        if(local_col == local_row)
                        {
                            // If diagonal type is non unit, do division by diagonal entry
                            // This is not required for unit diagonal for obvious reasons
                            if(diag_type == rocsparse_diag_type_non_unit)
                            {
                                diag     = j;
                                diag_val = static_cast<T>(1) / local_val;
                            }

                            break;
                        }

                        // Lower triangular part
                        rocsparse_int idx = (trans_X == rocsparse_operation_none)
                                                ? i * ldx + local_col
                                                : local_col * ldx + i;
                        sum               = std::fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = sum * diag_val;
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

template <typename T>
static inline void host_bsr_usolve(rocsparse_direction  dir,
                                   rocsparse_operation  trans_X,
                                   rocsparse_int        mb,
                                   rocsparse_int        nrhs,
                                   T                    alpha,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   const T*             bsr_val,
                                   rocsparse_int        bsr_dim,
                                   const T*             B,
                                   int64_t              ldb,
                                   T*                   X,
                                   int64_t              ldx,
                                   rocsparse_diag_type  diag_type,
                                   rocsparse_index_base base,
                                   rocsparse_int*       struct_pivot,
                                   rocsparse_int*       numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(rocsparse_int bsr_row = mb - 1; bsr_row >= 0; --bsr_row)
        {
            rocsparse_int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            rocsparse_int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            for(rocsparse_int bi = bsr_dim - 1; bi >= 0; --bi)
            {
                rocsparse_int local_row = bsr_row * bsr_dim + bi;

                int64_t idx_B = (trans_X == rocsparse_operation_none) ? i * ldb + local_row
                                                                      : local_row * ldb + i;
                int64_t idx_X = (trans_X == rocsparse_operation_none) ? i * ldx + local_row
                                                                      : local_row * ldx + i;
                T       sum   = alpha * B[idx_B];

                rocsparse_int diag     = -1;
                T             diag_val = static_cast<T>(0);

                for(rocsparse_int j = bsr_row_end - 1; j >= bsr_row_begin; --j)
                {
                    rocsparse_int bsr_col = bsr_col_ind[j] - base;

                    for(rocsparse_int bj = bsr_dim - 1; bj >= 0; --bj)
                    {
                        rocsparse_int local_col = bsr_col * bsr_dim + bj;
                        T             local_val = dir == rocsparse_direction_row
                                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        // Ignore all entries that are below the diagonal
                        if(local_col < local_row)
                        {
                            continue;
                        }

                        // Diagonal
                        if(local_col == local_row)
                        {
                            if(diag_type == rocsparse_diag_type_non_unit)
                            {
                                // Check for numerical zero
                                if(local_val == static_cast<T>(0))
                                {
                                    *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                                    local_val      = static_cast<T>(1);
                                }

                                diag     = j;
                                diag_val = static_cast<T>(1) / local_val;
                            }

                            continue;
                        }

                        // Upper triangular part
                        rocsparse_int idx = (trans_X == rocsparse_operation_none)
                                                ? i * ldx + local_col
                                                : local_col * ldx + i;
                        sum               = std::fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = sum * diag_val;
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

template <typename T>
void host_bsrsv(rocsparse_operation  trans,
                rocsparse_direction  dir,
                rocsparse_int        mb,
                rocsparse_int        nnzb,
                T                    alpha,
                const rocsparse_int* bsr_row_ptr,
                const rocsparse_int* bsr_col_ind,
                const T*             bsr_val,
                rocsparse_int        bsr_dim,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_lsolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_usolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> bsrt_row_ptr;
        std::vector<rocsparse_int> bsrt_col_ind;
        std::vector<T>             bsrt_val;

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_usolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_lsolve(dir,
                            rocsparse_operation_none,
                            mb,
                            1,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            x,
                            mb * bsr_dim,
                            y,
                            mb * bsr_dim,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                int64_t              nnz,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const A*             coo_val,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(trans == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            y[coo_row_ind[i] - base] = std::fma(static_cast<T>(alpha * coo_val[i]),
                                                static_cast<T>(x[coo_col_ind[i] - base]),
                                                static_cast<T>(y[coo_row_ind[i] - base]));
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            I row = coo_row_ind[i] - base;
            I col = coo_col_ind[i] - base;
            T val = (trans == rocsparse_operation_transpose) ? coo_val[i]
                                                             : rocsparse_conj(coo_val[i]);

            y[col] = std::fma(
                static_cast<T>(alpha * val), static_cast<T>(x[row]), static_cast<T>(y[col]));
        }
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv_aos(rocsparse_operation  trans,
                    I                    M,
                    I                    N,
                    int64_t              nnz,
                    T                    alpha,
                    const I*             coo_ind,
                    const A*             coo_val,
                    const X*             x,
                    T                    beta,
                    Y*                   y,
                    rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(trans)
    {
    case rocsparse_operation_none:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            y[coo_ind[2 * i] - base] = std::fma(static_cast<T>(alpha * coo_val[i]),
                                                static_cast<T>(x[coo_ind[2 * i + 1] - base]),
                                                static_cast<T>(y[coo_ind[2 * i] - base]));
        }

        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            I row = coo_ind[2 * i] - base;
            I col = coo_ind[2 * i + 1] - base;
            T val = (trans == rocsparse_operation_transpose) ? coo_val[i]
                                                             : rocsparse_conj(coo_val[i]);

            y[col] = std::fma(
                static_cast<T>(alpha * val), static_cast<T>(x[row]), static_cast<T>(y[col]));
        }

        break;
    }
    }
}

template <typename A>
inline A conj_val(A val, bool conj)
{
    return conj ? rocsparse_conj(val) : val;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
static void host_csrmv_general(rocsparse_operation  trans,
                               J                    M,
                               J                    N,
                               I                    nnz,
                               T                    alpha,
                               const I*             csr_row_ptr,
                               const J*             csr_col_ind,
                               const A*             csr_val,
                               const X*             x,
                               T                    beta,
                               Y*                   y,
                               rocsparse_index_base base,
                               rocsparse_spmv_alg   algo,
                               bool                 force_conj)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    if(trans == rocsparse_operation_none)
    {
        if(algo == rocsparse_spmv_alg_csr_rowsplit)
        {
            // Get device properties
            int             dev;
            hipDeviceProp_t prop;

            std::ignore = hipGetDevice(&dev);
            std::ignore = hipGetDeviceProperties(&prop, dev);

            int WF_SIZE;
            J   nnz_per_row = (M == 0) ? 0 : (nnz / M);

            if(nnz_per_row < 4)
                WF_SIZE = 2;
            else if(nnz_per_row < 8)
                WF_SIZE = 4;
            else if(nnz_per_row < 16)
                WF_SIZE = 8;
            else if(nnz_per_row < 32)
                WF_SIZE = 16;
            else if(nnz_per_row < 64 || prop.warpSize == 32)
                WF_SIZE = 32;
            else
                WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(J i = 0; i < M; ++i)
            {
                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                std::vector<T> sum(WF_SIZE, static_cast<T>(0));

                for(I j = row_begin; j < row_end; j += WF_SIZE)
                {
                    for(int k = 0; k < WF_SIZE; ++k)
                    {
                        if(j + k < row_end)
                        {
                            sum[k]
                                = std::fma(static_cast<T>(alpha * conj_val(csr_val[j + k], conj)),
                                           static_cast<T>(x[csr_col_ind[j + k] - base]),
                                           static_cast<T>(sum[k]));
                        }
                    }
                }

                for(int j = 1; j < WF_SIZE; j <<= 1)
                {
                    for(int k = 0; k < WF_SIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(beta == static_cast<T>(0))
                {
                    y[i] = sum[0];
                }
                else
                {
                    y[i] = std::fma(
                        static_cast<T>(beta), static_cast<T>(y[i]), static_cast<T>(sum[0]));
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(J i = 0; i < M; ++i)
            {
                T sum = static_cast<T>(0);
                T err = static_cast<T>(0);

                I row_begin = csr_row_ptr[i] - base;
                I row_end   = csr_row_ptr[i + 1] - base;

                for(I j = row_begin; j < row_end; ++j)
                {
                    T old  = sum;
                    T prod = alpha * conj_val(csr_val[j], conj) * x[csr_col_ind[j] - base];

                    sum = sum + prod;
                    err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
                }

                if(beta != static_cast<T>(0))
                {
                    y[i] = std::fma(
                        static_cast<T>(beta), static_cast<T>(y[i]), static_cast<T>(sum + err));
                }
                else
                {
                    y[i] = sum + err;
                }
            }
        }
    }
    else
    {
        // Scale y with beta
        for(J i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;
            T row_val   = alpha * x[i];

            for(I j = row_begin; j < row_end; ++j)
            {
                J col  = csr_col_ind[j] - base;
                A val  = conj_val(csr_val[j], conj);
                y[col] = std::fma(
                    static_cast<T>(val), static_cast<T>(row_val), static_cast<T>(y[col]));
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
static void host_csrmv_symmetric(rocsparse_operation  trans,
                                 J                    M,
                                 J                    N,
                                 I                    nnz,
                                 T                    alpha,
                                 const I*             csr_row_ptr,
                                 const J*             csr_col_ind,
                                 const A*             csr_val,
                                 const X*             x,
                                 T                    beta,
                                 Y*                   y,
                                 rocsparse_index_base base,
                                 rocsparse_spmv_alg   algo,
                                 bool                 force_conj)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    if(algo == rocsparse_spmv_alg_csr_rowsplit || trans != rocsparse_operation_none)
    {
        // Get device properties
        int             dev;
        hipDeviceProp_t prop;

        std::ignore = hipGetDevice(&dev);
        std::ignore = hipGetDeviceProperties(&prop, dev);

        int WF_SIZE;
        J   nnz_per_row = (M == 0) ? 0 : (nnz / M);

        if(nnz_per_row < 4)
            WF_SIZE = 2;
        else if(nnz_per_row < 8)
            WF_SIZE = 4;
        else if(nnz_per_row < 16)
            WF_SIZE = 8;
        else if(nnz_per_row < 32)
            WF_SIZE = 16;
        else if(nnz_per_row < 64 || prop.warpSize == 32)
            WF_SIZE = 32;
        else
            WF_SIZE = 64;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            std::vector<T> sum(WF_SIZE, static_cast<T>(0));

            for(I j = row_begin; j < row_end; j += WF_SIZE)
            {
                for(int k = 0; k < WF_SIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        A val  = conj_val(csr_val[j + k], conj);
                        sum[k] = std::fma(static_cast<T>(alpha * val),
                                          static_cast<T>(x[csr_col_ind[j + k] - base]),
                                          static_cast<T>(sum[k]));
                    }
                }
            }

            for(int j = 1; j < WF_SIZE; j <<= 1)
            {
                for(int k = 0; k < WF_SIZE - j; ++k)
                {
                    sum[k] += sum[k + j];
                }
            }

            if(beta == static_cast<T>(0))
            {
                y[i] = sum[0];
            }
            else
            {
                y[i] = std::fma(static_cast<T>(beta), static_cast<T>(y[i]), static_cast<T>(sum[0]));
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] = std::fma(static_cast<T>(conj_val(csr_val[j], conj)),
                                                        static_cast<T>(x_val),
                                                        static_cast<T>(y[csr_col_ind[j] - base]));
                }
            }
        }
    }
    else
    {
        // Scale y with beta
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; ++i)
        {
            y[i] *= beta;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            T sum = static_cast<T>(0);
            T err = static_cast<T>(0);

            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            for(I j = row_begin; j < row_end; ++j)
            {
                T old  = sum;
                T prod = alpha * conj_val(csr_val[j], conj) * x[csr_col_ind[j] - base];

                sum = sum + prod;
                err = (old - (sum - (sum - old))) + (prod - (sum - old)) + err;
            }

            y[i] += sum + err;
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            T x_val = alpha * x[i];
            for(I j = row_begin; j < row_end; ++j)
            {
                if((csr_col_ind[j] - base) != i)
                {
                    y[csr_col_ind[j] - base] = std::fma(static_cast<T>(conj_val(csr_val[j], conj)),
                                                        static_cast<T>(x_val),
                                                        static_cast<T>(y[csr_col_ind[j] - base]));
                }
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_csrmv(rocsparse_operation   trans,
                J                     M,
                J                     N,
                I                     nnz,
                T                     alpha,
                const I*              csr_row_ptr,
                const J*              csr_col_ind,
                const A*              csr_val,
                const X*              x,
                T                     beta,
                Y*                    y,
                rocsparse_index_base  base,
                rocsparse_matrix_type matrix_type,
                rocsparse_spmv_alg    algo,
                bool                  force_conj)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(matrix_type)
    {
    case rocsparse_matrix_type_symmetric:
    {
        host_csrmv_symmetric(trans,
                             M,
                             N,
                             nnz,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             x,
                             beta,
                             y,
                             base,
                             algo,
                             force_conj);
        break;
    }

    case rocsparse_matrix_type_general:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        host_csrmv_general(trans,
                           M,
                           N,
                           nnz,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           base,
                           algo,
                           force_conj);
        break;
    }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_cscmv(rocsparse_operation trans,
                J                   M,
                J                   N,
                I                   nnz,
                T                   alpha,
                const I* __restrict csc_col_ptr,
                const J* __restrict csc_row_ind,
                const A* __restrict csc_val,
                const X* __restrict x,
                T beta,
                Y* __restrict y,
                rocsparse_index_base  base,
                rocsparse_matrix_type matrix_type,
                rocsparse_spmv_alg    algo)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(trans)
    {
    case rocsparse_operation_none:
    {
        return host_csrmv(rocsparse_operation_transpose,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          false);
    }
    case rocsparse_operation_transpose:
    {
        return host_csrmv(rocsparse_operation_none,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          false);
    }
    case rocsparse_operation_conjugate_transpose:
    {
        return host_csrmv(rocsparse_operation_none,
                          N,
                          M,
                          nnz,
                          alpha,
                          csc_col_ptr,
                          csc_row_ind,
                          csc_val,
                          x,
                          beta,
                          y,
                          base,
                          matrix_type,
                          algo,
                          true);
    }
    }
}

template <typename I, typename J, typename T>
static void host_csr_lsolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            int64_t              x_inc,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            J*                   struct_pivot,
                            J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process lower triangular part
    for(J row = 0; row < M; ++row)
    {
        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_begin; l < row_end; l += prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocsparse_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0
                    // and store index for later use.
                    *numeric_pivot = std::min(*numeric_pivot, row + base);
                    local_val      = static_cast<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    break;
                }

                // Lower triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
static void host_csr_usolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            int64_t              x_inc,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            J*                   struct_pivot,
                            J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process upper triangular part
    for(J row = M - 1; row >= 0; --row)
    {

        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(local_val == static_cast<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, row + base);
                            local_val      = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    continue;
                }

                // Upper triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
void host_csrsv(rocsparse_operation  trans,
                J                    M,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                const T*             x,
                int64_t              x_inc,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_lsolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_usolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == rocsparse_operation_transpose
            || trans == rocsparse_operation_conjugate_transpose)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        rocsparse_action_numeric,
                        base);

        if(trans == rocsparse_operation_conjugate_transpose)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = rocsparse_conj(csrt_val[i]);
            }
        }

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_usolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_lsolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_coosv(rocsparse_operation  trans,
                I                    M,
                int64_t              nnz,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                I*                   struct_pivot,
                I*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        std::vector<int32_t> csr_row_ptr(M + 1);

        host_coo_to_csr<int32_t, I>(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsv<int32_t, I>(trans,
                               M,
                               nnz,
                               alpha,
                               csr_row_ptr.data(),
                               coo_col_ind,
                               coo_val,
                               x,
                               (int64_t)1,
                               y,
                               diag_type,
                               fill_mode,
                               base,
                               struct_pivot,
                               numeric_pivot);
    }
    else
    {
        std::vector<int64_t> csr_row_ptr(M + 1);

        host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsv(trans,
                   M,
                   nnz,
                   alpha,
                   csr_row_ptr.data(),
                   coo_col_ind,
                   coo_val,
                   x,
                   (int64_t)1,
                   y,
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
void host_ellmv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                T                    alpha,
                const I*             ell_col_ind,
                const A*             ell_val,
                I                    ell_width,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(trans == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I i = 0; i < M; ++i)
        {
            T sum = static_cast<T>(0);
            for(I p = 0; p < ell_width; ++p)
            {
                int64_t idx = (int64_t)p * M + i;
                I       col = ell_col_ind[idx] - base;

                if(col >= 0 && col < N)
                {
                    sum = std::fma(
                        static_cast<T>(ell_val[idx]), static_cast<T>(x[col]), static_cast<T>(sum));
                }
                else
                {
                    break;
                }
            }

            if(beta != static_cast<T>(0))
            {
                y[i] = std::fma(
                    static_cast<T>(beta), static_cast<T>(y[i]), static_cast<T>(alpha * sum));
            }
            else
            {
                y[i] = alpha * sum;
            }
        }
    }
    else
    {
        // Scale y with beta
        for(I i = 0; i < N; ++i)
        {
            y[i] *= beta;
        }

        // Transposed SpMV
        for(I i = 0; i < M; ++i)
        {
            T row_val = alpha * x[i];

            for(I p = 0; p < ell_width; ++p)
            {
                int64_t idx = (int64_t)p * M + i;
                I       col = ell_col_ind[idx] - base;

                if(col >= 0 && col < N)
                {
                    T val = (trans == rocsparse_operation_conjugate_transpose)
                                ? rocsparse_conj(ell_val[idx])
                                : ell_val[idx];

                    y[col] = std::fma(
                        static_cast<T>(val), static_cast<T>(row_val), static_cast<T>(y[col]));
                }
                else
                {
                    break;
                }
            }
        }
    }
}

template <typename T>
void host_hybmv(rocsparse_operation  trans,
                rocsparse_int        M,
                rocsparse_int        N,
                T                    alpha,
                rocsparse_int        ell_nnz,
                const rocsparse_int* ell_col_ind,
                const T*             ell_val,
                rocsparse_int        ell_width,
                rocsparse_int        coo_nnz,
                const rocsparse_int* coo_row_ind,
                const rocsparse_int* coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(M == 0 || N == 0)
    {
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? M : N;

        if(ysize > 0)
        {
            for(rocsparse_int i = 0; i < ysize; i++)
            {
                y[i] = beta * y[i];
            }
        }

        return;
    }

    T coo_beta = beta;

    // ELL part
    if(ell_nnz > 0)
    {
        host_ellmv(trans, M, N, alpha, ell_col_ind, ell_val, ell_width, x, beta, y, base);
        coo_beta = static_cast<T>(1);
    }

    // COO part
    if(coo_nnz > 0)
    {
        host_coomv(
            trans, M, N, coo_nnz, alpha, coo_row_ind, coo_col_ind, coo_val, x, coo_beta, y, base);
    }
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_bsrmm(rocsparse_handle     handle,
                rocsparse_direction  dir,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                J                    Mb,
                J                    N,
                J                    Kb,
                I                    nnzb,
                T                    alpha,
                const A*             bsr_val_A,
                const I*             bsr_row_ptr_A,
                const J*             bsr_col_ind_A,
                J                    block_dim,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(transA != rocsparse_operation_none)
    {
        return;
    }

    if(transB != rocsparse_operation_none && transB != rocsparse_operation_transpose)
    {
        return;
    }

    J M = Mb * block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; i++)
    {
        J local_row = i % block_dim;

        I row_begin = bsr_row_ptr_A[i / block_dim] - base;
        I row_end   = bsr_row_ptr_A[i / block_dim + 1] - base;

        for(J j = 0; j < N; j++)
        {
            int64_t idx_C = (order_C == rocsparse_order_column) ? i + j * ldc : i * ldc + j;

            T sum = static_cast<T>(0);

            for(I s = row_begin; s < row_end; s++)
            {
                for(J t = 0; t < block_dim; t++)
                {
                    int64_t idx_A = (dir == rocsparse_direction_row)
                                        ? block_dim * block_dim * s + block_dim * local_row + t
                                        : block_dim * block_dim * s + block_dim * t + local_row;

                    int64_t idx_B = 0;
                    if((transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                       || (transB == rocsparse_operation_transpose
                           && order_B == rocsparse_order_row)
                       || (transB == rocsparse_operation_conjugate_transpose
                           && order_B == rocsparse_order_row))
                    {
                        idx_B = j * ldb + block_dim * (bsr_col_ind_A[s] - base) + t;
                    }
                    else
                    {
                        idx_B = (block_dim * (bsr_col_ind_A[s] - base) + t) * ldb + j;
                    }

                    sum = std::fma(static_cast<T>(bsr_val_A[idx_A]),
                                   static_cast<T>(dense_B[idx_B]),
                                   static_cast<T>(sum));
                }
            }

            if(beta == static_cast<T>(0))
            {
                dense_C[idx_C] = alpha * sum;
            }
            else
            {
                dense_C[idx_C] = std::fma(static_cast<T>(beta),
                                          static_cast<T>(dense_C[idx_C]),
                                          static_cast<T>(alpha * sum));
            }
        }
    }
}

template <typename T>
void host_gebsrmm(rocsparse_handle          handle,
                  rocsparse_direction       dir,
                  rocsparse_operation       transA,
                  rocsparse_operation       transB,
                  rocsparse_int             Mb,
                  rocsparse_int             N,
                  rocsparse_int             Kb,
                  rocsparse_int             nnzb,
                  const T*                  alpha,
                  const rocsparse_mat_descr descr,
                  const T*                  bsr_val_A,
                  const rocsparse_int*      bsr_row_ptr_A,
                  const rocsparse_int*      bsr_col_ind_A,
                  rocsparse_int             row_block_dim,
                  rocsparse_int             col_block_dim,
                  const T*                  B,
                  int64_t                   ldb,
                  const T*                  beta,
                  T*                        C,
                  int64_t                   ldc)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(transA != rocsparse_operation_none)
    {
        return;
    }

    if(transB != rocsparse_operation_none && transB != rocsparse_operation_transpose)
    {
        return;
    }
    rocsparse_index_base base = rocsparse_get_mat_index_base(descr);

    rocsparse_int M = Mb * row_block_dim;

    const rocsparse_int rowXcol_block_dim = row_block_dim * col_block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int row_idx = 0; row_idx < M; ++row_idx)
    {
        const rocsparse_int row_block_idx = row_idx / row_block_dim,
                            row_local_idx = row_idx % row_block_dim;

        const rocsparse_int start = bsr_row_ptr_A[row_block_idx] - base,
                            bound = bsr_row_ptr_A[row_block_idx + 1] - base;

        for(rocsparse_int col_idx = 0; col_idx < N; ++col_idx)
        {
            const int64_t idx_C = ldc * col_idx + row_idx;

            T sum = static_cast<T>(0);

            for(rocsparse_int at = start; at < bound; ++at)
            {
                for(rocsparse_int col_local_idx = 0; col_local_idx < col_block_dim; ++col_local_idx)
                {
                    const int64_t idx_A = (dir == rocsparse_direction_row)
                                              ? rowXcol_block_dim * at
                                                    + col_block_dim * row_local_idx + col_local_idx
                                              : rowXcol_block_dim * at
                                                    + row_block_dim * col_local_idx + row_local_idx;

                    const int64_t idx_B
                        = (transB == rocsparse_operation_none)
                              ? col_idx * ldb + col_block_dim * (bsr_col_ind_A[at] - base)
                                    + col_local_idx
                              : (col_block_dim * (bsr_col_ind_A[at] - base) + col_local_idx) * ldb
                                    + col_idx;

                    sum = std::fma(bsr_val_A[idx_A], B[idx_B], sum);
                }
            }

            if(*beta == static_cast<T>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_csrmm(J                    M,
                J                    N,
                J                    K,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr_A,
                const J*             csr_col_ind_A,
                const A*             csr_val_A,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base,
                bool                 force_conj_A)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bool conj_A = (transA == rocsparse_operation_conjugate_transpose || force_conj_A);
    bool conj_B = (transB == rocsparse_operation_conjugate_transpose);

    if(transA == rocsparse_operation_none)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                for(I k = row_begin; k < row_end; ++k)
                {
                    int64_t idx_B = 0;
                    if((transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                       || (transB == rocsparse_operation_transpose
                           && order_B == rocsparse_order_row)
                       || (transB == rocsparse_operation_conjugate_transpose
                           && order_B == rocsparse_order_row))
                    {
                        idx_B = (csr_col_ind_A[k] - base + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + (csr_col_ind_A[k] - base) * ldb);
                    }

                    sum = std::fma(static_cast<T>(conj_val(csr_val_A[k], conj_A)),
                                   static_cast<T>(conj_val(dense_B[idx_B], conj_B)),
                                   static_cast<T>(sum));
                }

                int64_t idx_C = (order_C == rocsparse_order_column) ? i + j * ldc : i * ldc + j;

                if(beta == static_cast<T>(0))
                {
                    dense_C[idx_C] = alpha * sum;
                }
                else
                {
                    dense_C[idx_C] = std::fma(static_cast<T>(beta),
                                              static_cast<T>(dense_C[idx_C]),
                                              static_cast<T>(alpha * sum));
                }
            }
        }
    }
    else
    {
        // scale C by beta
        for(J i = 0; i < K; i++)
        {
            for(J j = 0; j < N; ++j)
            {
                int64_t idx_C  = (order_C == rocsparse_order_column) ? i + j * ldc : i * ldc + j;
                dense_C[idx_C] = beta * dense_C[idx_C];
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                for(I k = row_begin; k < row_end; ++k)
                {
                    J col = csr_col_ind_A[k] - base;
                    T val = conj_val(csr_val_A[k], conj_A);

                    int64_t idx_B = 0;

                    if((transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                       || (transB == rocsparse_operation_transpose
                           && order_B == rocsparse_order_row)
                       || (transB == rocsparse_operation_conjugate_transpose
                           && order_B == rocsparse_order_row))
                    {
                        idx_B = (i + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + i * ldb);
                    }

                    int64_t idx_C
                        = (order_C == rocsparse_order_column) ? col + j * ldc : col * ldc + j;

                    dense_C[idx_C] += alpha * val * conj_val(dense_B[idx_B], conj_B);
                }
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_csrmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        int64_t              offsets_batch_stride_A,
                        int64_t              columns_values_batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             csr_row_ptr_A,
                        const J*             csr_col_ind_A,
                        const A*             csr_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        J                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        J                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base,
                        bool                 force_conj_A)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    const bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    const bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_val_A,
                       dense_B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_B)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       dense_B,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       dense_B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
}

template <typename T, typename I, typename A, typename B, typename C>
void host_coomm(I                    M,
                I                    N,
                I                    K,
                int64_t              nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind_A,
                const I*             coo_col_ind_A,
                const A*             coo_val_A,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bool conj_A = (transA == rocsparse_operation_conjugate_transpose);
    bool conj_B = (transB == rocsparse_operation_conjugate_transpose);

    if(transA == rocsparse_operation_none)
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < M; ++i)
            {
                int64_t idx_C = (order_C == rocsparse_order_column) ? i + j * ldc : i * ldc + j;
                dense_C[idx_C] *= beta;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = alpha * coo_val_A[i];

                int64_t idx_C = (order_C == rocsparse_order_column) ? row + j * ldc : row * ldc + j;

                int64_t idx_B = 0;
                if((transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                   || (transB != rocsparse_operation_none && order_B != rocsparse_order_column))
                {
                    idx_B = (col + j * ldb);
                }
                else
                {
                    idx_B = (j + col * ldb);
                }

                dense_C[idx_C] = std::fma(static_cast<T>(val),
                                          static_cast<T>(conj_val(dense_B[idx_B], conj_B)),
                                          static_cast<T>(dense_C[idx_C]));
            }
        }
    }
    else
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < K; ++i)
            {
                int64_t idx_C = (order_C == rocsparse_order_column) ? i + j * ldc : i * ldc + j;
                dense_C[idx_C] *= beta;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = alpha * conj_val(coo_val_A[i], conj_A);

                int64_t idx_C = (order_C == rocsparse_order_column) ? col + j * ldc : col * ldc + j;

                int64_t idx_B = 0;
                if((transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                   || (transB != rocsparse_operation_none && order_B != rocsparse_order_column))
                {
                    idx_B = (row + j * ldb);
                }
                else
                {
                    idx_B = (j + row * ldb);
                }

                dense_C[idx_C] = std::fma(static_cast<T>(val),
                                          static_cast<T>(conj_val(dense_B[idx_B], conj_B)),
                                          static_cast<T>(dense_C[idx_C]));
            }
        }
    }
}

template <typename T, typename I, typename A, typename B, typename C>
void host_coomm_batched(I                    M,
                        I                    N,
                        I                    K,
                        int64_t              nnz,
                        I                    batch_count_A,
                        int64_t              batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             coo_row_ind_A,
                        const I*             coo_col_ind_A,
                        const A*             coo_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        I                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        I                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A,
                       coo_col_ind_A,
                       coo_val_A,
                       dense_B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_B)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       dense_B,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       dense_B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       dense_C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_cscmm(J                   M,
                J                   N,
                J                   K,
                rocsparse_operation transA,
                rocsparse_operation transB,
                T                   alpha,
                const I* __restrict csc_col_ptr_A,
                const J* __restrict csc_row_ind_A,
                const A* __restrict csc_val_A,
                const B* __restrict dense_B,
                int64_t         ldb,
                rocsparse_order order_B,
                T               beta,
                C* __restrict dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(transA)
    {
    case rocsparse_operation_none:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocsparse_operation_transpose,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          dense_B,
                          ldb,
                          order_B,
                          beta,
                          dense_C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case rocsparse_operation_transpose:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocsparse_operation_none,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          dense_B,
                          ldb,
                          order_B,
                          beta,
                          dense_C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case rocsparse_operation_conjugate_transpose:
    {
        return host_csrmm(K,
                          N,
                          M,
                          rocsparse_operation_none,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          dense_B,
                          ldb,
                          order_B,
                          beta,
                          dense_C,
                          ldc,
                          order_C,
                          base,
                          true);
    }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_cscmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        int64_t              offsets_batch_stride_A,
                        int64_t              rows_values_batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             csc_col_ptr_A,
                        const J*             csc_row_ind_A,
                        const A*             csc_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        J                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        J                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(transA)
    {
    case rocsparse_operation_none:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocsparse_operation_transpose,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  dense_B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  dense_C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case rocsparse_operation_transpose:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocsparse_operation_none,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  dense_B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  dense_C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case rocsparse_operation_conjugate_transpose:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  rocsparse_operation_none,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  dense_B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  dense_C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  true);
    }
    }
}

template <typename I, typename J, typename T>
static inline void host_lssolve(J                    M,
                                J                    nrhs,
                                rocsparse_operation  transB,
                                T                    alpha,
                                const I*             csr_row_ptr,
                                const J*             csr_col_ind,
                                const T*             csr_val,
                                T*                   B,
                                int64_t              ldb,
                                rocsparse_order      order_B,
                                rocsparse_diag_type  diag_type,
                                rocsparse_index_base base,
                                J*                   struct_pivot,
                                J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(J row = 0; row < M; ++row)
        {
            int64_t idx_B
                = (transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                      ? i * ldb + row
                      : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocsparse_operation_conjugate_transpose)
            {
                sum = alpha * rocsparse_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_begin; j < row_end; ++j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocsparse_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0 and store
                    // index for later use
                    *numeric_pivot = std::min(*numeric_pivot, row + base);
                    local_val      = static_cast<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    break;
                }

                // Lower triangular part
                int64_t idx
                    = (transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                          ? i * ldb + local_col
                          : local_col * ldb + i;

                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename I, typename J, typename T>
static inline void host_ussolve(J                    M,
                                J                    nrhs,
                                rocsparse_operation  transB,
                                T                    alpha,
                                const I*             csr_row_ptr,
                                const J*             csr_col_ind,
                                const T*             csr_val,
                                T*                   B,
                                int64_t              ldb,
                                rocsparse_order      order_B,
                                rocsparse_diag_type  diag_type,
                                rocsparse_index_base base,
                                J*                   struct_pivot,
                                J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(J row = M - 1; row >= 0; --row)
        {
            int64_t idx_B
                = (transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                      ? i * ldb + row
                      : row * ldb + i;

            T sum = static_cast<T>(0);
            if(transB == rocsparse_operation_conjugate_transpose)
            {
                sum = alpha * rocsparse_conj(B[idx_B]);
            }
            else
            {
                sum = alpha * B[idx_B];
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = static_cast<T>(0);

            for(I j = row_end - 1; j >= row_begin; --j)
            {
                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(local_val == static_cast<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, row + base);
                            local_val      = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    continue;
                }

                // Upper triangular part
                int64_t idx
                    = (transB == rocsparse_operation_none && order_B == rocsparse_order_column)
                          ? i * ldb + local_col
                          : local_col * ldb + i;

                sum = std::fma(-local_val, B[idx], sum);
            }

            if(diag_type == rocsparse_diag_type_non_unit)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = sum * diag_val;
            }
            else
            {
                B[idx_B] = sum;
            }
        }
    }
}

template <typename I, typename J, typename T>
void host_csrsm(J                    M,
                J                    nrhs,
                I                    nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                T*                   B,
                int64_t              ldb,
                rocsparse_order      order_B,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(nrhs == 0)
    {
        *struct_pivot  = M + 1;
        *numeric_pivot = M + 1;
        *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);
        *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
        *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
    }
    else if(nrhs == 1)
    {
        J B_m = (transB == rocsparse_operation_none) ? M : nrhs;
        J B_n = (transB == rocsparse_operation_none) ? nrhs : M;

        int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
        int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;

        std::vector<T> x(nrowB * ncolB, 0);
        std::vector<T> y(M, 0);
        for(size_t i = 0; i < x.size(); i++)
        {
            x[i] = B[i];
        }

        for(J i = 0; i < M; i++)
        {
            y[i] = B[i];
        }

        int64_t x_inc
            = (transB == rocsparse_operation_none && order_B == rocsparse_order_column) ? 1 : ldb;

        host_csrsv(transA,
                   M,
                   nnz,
                   alpha,
                   csr_row_ptr,
                   csr_col_ind,
                   csr_val,
                   x.data(),
                   x_inc,
                   y.data(),
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);

        if((transB == rocsparse_operation_none && order_B == rocsparse_order_column))
        {
            for(J i = 0; i < M; i++)
            {
                B[i] = y[i];
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                B[i * ldb] = y[i];
            }
        }
    }
    else
    {
        // Initialize pivot
        *struct_pivot  = M + 1;
        *numeric_pivot = M + 1;

        if(transA == rocsparse_operation_none)
        {
            if(fill_mode == rocsparse_fill_mode_lower)
            {
                host_lssolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
            else
            {
                host_ussolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
        }
        else if(transA == rocsparse_operation_transpose
                || transA == rocsparse_operation_conjugate_transpose)
        {
            // Transpose matrix
            std::vector<I> csrt_row_ptr(M + 1);
            std::vector<J> csrt_col_ind(nnz);
            std::vector<T> csrt_val(nnz);

            host_csr_to_csc<I, J, T>(M,
                                     M,
                                     nnz,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csrt_col_ind,
                                     csrt_row_ptr,
                                     csrt_val,
                                     rocsparse_action_numeric,
                                     base);

            if(transA == rocsparse_operation_conjugate_transpose)
            {
                for(size_t i = 0; i < csrt_val.size(); i++)
                {
                    csrt_val[i] = rocsparse_conj(csrt_val[i]);
                }
            }

            if(fill_mode == rocsparse_fill_mode_lower)
            {
                host_ussolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csrt_row_ptr.data(),
                             csrt_col_ind.data(),
                             csrt_val.data(),
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
            else
            {
                host_lssolve(M,
                             nrhs,
                             transB,
                             alpha,
                             csrt_row_ptr.data(),
                             csrt_col_ind.data(),
                             csrt_val.data(),
                             B,
                             ldb,
                             order_B,
                             diag_type,
                             base,
                             struct_pivot,
                             numeric_pivot);
            }
        }

        *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

        *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
        *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
    }
}

template <typename I, typename T>
void host_coosm(I                    M,
                I                    nrhs,
                int64_t              nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                T*                   B,
                int64_t              ldb,
                rocsparse_order      order_B,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                I*                   struct_pivot,
                I*                   numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        std::vector<int32_t> csr_row_ptr(M + 1);

        host_coo_to_csr<int32_t, I>(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsm<int32_t, I>(M,
                               nrhs,
                               nnz,
                               transA,
                               transB,
                               alpha,
                               csr_row_ptr.data(),
                               coo_col_ind,
                               coo_val,
                               B,
                               ldb,
                               order_B,
                               diag_type,
                               fill_mode,
                               base,
                               struct_pivot,
                               numeric_pivot);
    }
    else
    {
        std::vector<int64_t> csr_row_ptr(M + 1);

        host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        host_csrsm(M,
                   nrhs,
                   nnz,
                   transA,
                   transB,
                   alpha,
                   csr_row_ptr.data(),
                   coo_col_ind,
                   coo_val,
                   B,
                   ldb,
                   order_B,
                   diag_type,
                   fill_mode,
                   base,
                   struct_pivot,
                   numeric_pivot);
    }
}

template <typename T>
void host_bsrsm(rocsparse_int       mb,
                rocsparse_int       nrhs,
                rocsparse_int       nnzb,
                rocsparse_direction dir,
                rocsparse_operation transA,
                rocsparse_operation transX,
                T                   alpha,
                const rocsparse_int* __restrict bsr_row_ptr,
                const rocsparse_int* __restrict bsr_col_ind,
                const T* __restrict bsr_val,
                rocsparse_int        bsr_dim,
                const T*             B,
                int64_t              ldb,
                T*                   X,
                int64_t              ldx,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(transA == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_lsolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_usolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_val,
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(transA == rocsparse_operation_transpose)
    {
        // Transpose matrix
        std::vector<rocsparse_int> bsrt_row_ptr(mb + 1);
        std::vector<rocsparse_int> bsrt_col_ind(nnzb);
        std::vector<T>             bsrt_val(nnzb * bsr_dim * bsr_dim);

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_bsr_usolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_bsr_lsolve(dir,
                            transX,
                            mb,
                            nrhs,
                            alpha,
                            bsrt_row_ptr.data(),
                            bsrt_col_ind.data(),
                            bsrt_val.data(),
                            bsr_dim,
                            B,
                            ldb,
                            X,
                            ldx,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_gemvi(I                    M,
                I                    N,
                T                    alpha,
                const T*             A,
                int64_t              lda,
                I                    nnz,
                const T*             x_val,
                const I*             x_ind,
                T                    beta,
                T*                   y,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I i = 0; i < M; ++i)
    {
        T sum = static_cast<T>(0);

        for(I j = 0; j < nnz; ++j)
        {
            sum = std::fma(x_val[j], A[(x_ind[j] - base) * lda + i], sum);
        }

        y[i] = std::fma(alpha, sum, beta * y[i]);
    }
}

template <typename T>
void host_gemmi(rocsparse_int        M,
                rocsparse_int        N,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const T*             A,
                int64_t              lda,
                const rocsparse_int* csr_row_ptr,
                const rocsparse_int* csr_col_ind,
                const T*             csr_val,
                T                    beta,
                T*                   C,
                int64_t              ldc,
                rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(transB == rocsparse_operation_transpose)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                T sum = static_cast<T>(0);

                rocsparse_int row_begin = csr_row_ptr[j] - base;
                rocsparse_int row_end   = csr_row_ptr[j + 1] - base;

                for(rocsparse_int k = row_begin; k < row_end; ++k)
                {
                    rocsparse_int col_B = csr_col_ind[k] - base;
                    T             val_B = csr_val[k];
                    T             val_A = A[col_B * lda + i];

                    sum = std::fma(val_A, val_B, sum);
                }

                C[j * ldc + i] = std::fma(beta, C[j * ldc + i], alpha * sum);
            }
        }
    }
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J>
void host_bsrgemm_nnzb(J                    Mb,
                       J                    Nb,
                       J                    Kb,
                       J                    block_dim,
                       const T*             alpha,
                       const I*             bsr_row_ptr_A,
                       const J*             bsr_col_ind_A,
                       const I*             bsr_row_ptr_B,
                       const J*             bsr_col_ind_B,
                       const T*             beta,
                       const I*             bsr_row_ptr_D,
                       const J*             bsr_col_ind_D,
                       I*                   bsr_row_ptr_C,
                       I*                   nnzb_C,
                       rocsparse_index_base base_A,
                       rocsparse_index_base base_B,
                       rocsparse_index_base base_C,
                       rocsparse_index_base base_D)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return host_csrgemm_nnz(Mb,
                            Nb,
                            Kb,
                            alpha,
                            bsr_row_ptr_A,
                            bsr_col_ind_A,
                            bsr_row_ptr_B,
                            bsr_col_ind_B,
                            beta,
                            bsr_row_ptr_D,
                            bsr_col_ind_D,
                            bsr_row_ptr_C,
                            nnzb_C,
                            base_A,
                            base_B,
                            base_C,
                            base_D);
}

template <typename T, typename I, typename J>
void host_bsrgemm(rocsparse_direction  dir,
                  J                    Mb,
                  J                    Nb,
                  J                    Kb,
                  J                    block_dim,
                  const T*             alpha,
                  const I*             bsr_row_ptr_A,
                  const J*             bsr_col_ind_A,
                  const T*             bsr_val_A,
                  const I*             bsr_row_ptr_B,
                  const J*             bsr_col_ind_B,
                  const T*             bsr_val_B,
                  const T*             beta,
                  const I*             bsr_row_ptr_D,
                  const J*             bsr_col_ind_D,
                  const T*             bsr_val_D,
                  const I*             bsr_row_ptr_C,
                  J*                   bsr_col_ind_C,
                  T*                   bsr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C,
                  rocsparse_index_base base_D)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(Mb == 0 || Nb == 0)
    {
        return;
    }
    else if(alpha && !beta && (Kb == 0))
    {
        return;
    }
    else if(!alpha && !beta)
    {
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<I> nnzb(Nb, -1);

        int nthreads = 1;
        int tid      = 0;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid      = omp_get_thread_num();
#endif

        J rows_per_thread = (Mb + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, Mb);

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            I row_begin_C = bsr_row_ptr_C[i] - base_C;
            I row_end_C   = row_begin_C;

            if(alpha)
            {
                I row_begin_A = bsr_row_ptr_A[i] - base_A;
                I row_end_A   = bsr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = bsr_col_ind_A[j] - base_A;

                    I row_begin_B = bsr_row_ptr_B[col_A] - base_B;
                    I row_end_B   = bsr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(I k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        J col_B = bsr_col_ind_B[k] - base_B;

                        // Check if a new nnzb is generated or if the product is appended
                        if(nnzb[col_B] < row_begin_C)
                        {
                            nnzb[col_B]              = row_end_C;
                            bsr_col_ind_C[row_end_C] = col_B + base_C;

                            for(J r = 0; r < block_dim; r++)
                            {
                                for(J c = 0; c < block_dim; c++)
                                {
                                    T val_C = static_cast<T>(0);

                                    if(dir == rocsparse_direction_row)
                                    {
                                        for(J a = 0; a < block_dim; a++)
                                        {
                                            val_C = std::fma(bsr_val_A[block_dim * block_dim * j
                                                                       + block_dim * r + a],
                                                             bsr_val_B[block_dim * block_dim * k
                                                                       + block_dim * a + c],
                                                             val_C);
                                        }

                                        bsr_val_C[block_dim * block_dim * row_end_C + block_dim * r
                                                  + c]
                                            = *alpha * val_C;
                                    }
                                    else
                                    {
                                        for(J a = 0; a < block_dim; a++)
                                        {
                                            val_C = std::fma(bsr_val_A[block_dim * block_dim * j
                                                                       + block_dim * a + r],
                                                             bsr_val_B[block_dim * block_dim * k
                                                                       + block_dim * c + a],
                                                             val_C);
                                        }

                                        bsr_val_C[block_dim * block_dim * row_end_C + block_dim * c
                                                  + r]
                                            = *alpha * val_C;
                                    }
                                }
                            }

                            ++row_end_C;
                        }
                        else
                        {
                            for(J r = 0; r < block_dim; r++)
                            {
                                for(J c = 0; c < block_dim; c++)
                                {
                                    T val_C = static_cast<T>(0);

                                    if(dir == rocsparse_direction_row)
                                    {
                                        for(J a = 0; a < block_dim; a++)
                                        {
                                            val_C = std::fma(bsr_val_A[block_dim * block_dim * j
                                                                       + block_dim * r + a],
                                                             bsr_val_B[block_dim * block_dim * k
                                                                       + block_dim * a + c],
                                                             val_C);
                                        }

                                        bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                  + block_dim * r + c]
                                            = std::fma(*alpha,
                                                       val_C,
                                                       bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                                 + block_dim * r + c]);
                                    }
                                    else
                                    {
                                        for(J a = 0; a < block_dim; a++)
                                        {
                                            val_C = std::fma(bsr_val_A[block_dim * block_dim * j
                                                                       + block_dim * a + r],
                                                             bsr_val_B[block_dim * block_dim * k
                                                                       + block_dim * c + a],
                                                             val_C);
                                        }

                                        bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                  + block_dim * c + r]
                                            = std::fma(*alpha,
                                                       val_C,
                                                       bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                                 + block_dim * c + r]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add nnzb of D if beta != 0
            if(beta)
            {
                I row_begin_D = bsr_row_ptr_D[i] - base_D;
                I row_end_D   = bsr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    J col_D = bsr_col_ind_D[j] - base_D;
                    // Current value of D

                    // Check if a new nnzb is generated or if the value is added
                    if(nnzb[col_D] < row_begin_C)
                    {
                        nnzb[col_D] = row_end_C;

                        bsr_col_ind_C[row_end_C] = col_D + base_C;

                        for(J r = 0; r < block_dim; r++)
                        {
                            for(J c = 0; c < block_dim; c++)
                            {
                                if(dir == rocsparse_direction_row)
                                {
                                    bsr_val_C[block_dim * block_dim * row_end_C + block_dim * r + c]
                                        = *beta
                                          * bsr_val_D[block_dim * block_dim * j + block_dim * r
                                                      + c];
                                }
                                else
                                {
                                    bsr_val_C[block_dim * block_dim * row_end_C + block_dim * c + r]
                                        = *beta
                                          * bsr_val_D[block_dim * block_dim * j + block_dim * c
                                                      + r];
                                }
                            }
                        }

                        ++row_end_C;
                    }
                    else
                    {
                        for(J r = 0; r < block_dim; r++)
                        {
                            for(J c = 0; c < block_dim; c++)
                            {
                                if(dir == rocsparse_direction_row)
                                {
                                    bsr_val_C[block_dim * block_dim * nnzb[col_D] + block_dim * r
                                              + c]
                                        = std::fma(*beta,
                                                   bsr_val_D[block_dim * block_dim * j
                                                             + block_dim * r + c],
                                                   bsr_val_C[block_dim * block_dim * nnzb[col_D]
                                                             + block_dim * r + c]);
                                }
                                else
                                {
                                    bsr_val_C[block_dim * block_dim * nnzb[col_D] + block_dim * c
                                              + r]
                                        = std::fma(*beta,
                                                   bsr_val_D[block_dim * block_dim * j
                                                             + block_dim * c + r],
                                                   bsr_val_C[block_dim * block_dim * nnzb[col_D]
                                                             + block_dim * c + r]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    I nnzb = bsr_row_ptr_C[Mb] - base_C;

    std::vector<J> col(nnzb);
    std::vector<T> val(block_dim * block_dim * nnzb);

    memcpy(col.data(), bsr_col_ind_C, sizeof(J) * nnzb);
    memcpy(val.data(), bsr_val_C, sizeof(T) * block_dim * block_dim * nnzb);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < Mb; ++i)
    {
        I row_begin = bsr_row_ptr_C[i] - base_C;
        I row_end   = bsr_row_ptr_C[i + 1] - base_C;
        J row_nnzb  = row_end - row_begin;

        std::vector<J> perm(row_nnzb);
        for(J j = 0; j < row_nnzb; ++j)
        {
            perm[j] = j;
        }

        J* col_entry = col.data() + row_begin;
        T* val_entry = val.data() + block_dim * block_dim * row_begin;

        std::sort(perm.begin(), perm.end(), [&](const J& a, const J& b) {
            return col_entry[a] < col_entry[b];
        });

        for(J j = 0; j < row_nnzb; ++j)
        {
            bsr_col_ind_C[row_begin + j] = col_entry[perm[j]];

            for(J r = 0; r < block_dim; r++)
            {
                for(J c = 0; c < block_dim; c++)
                {
                    if(dir == rocsparse_direction_row)
                    {
                        bsr_val_C[block_dim * block_dim * (row_begin + j) + block_dim * r + c]
                            = val_entry[block_dim * block_dim * perm[j] + block_dim * r + c];
                    }
                    else
                    {
                        bsr_val_C[block_dim * block_dim * (row_begin + j) + block_dim * c + r]
                            = val_entry[block_dim * block_dim * perm[j] + block_dim * c + r];
                    }
                }
            }
        }
    }
}

template <typename T>
void host_bsrgeam_nnzb(rocsparse_direction  dir,
                       rocsparse_int        Mb,
                       rocsparse_int        Nb,
                       rocsparse_int        block_dim,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr_A,
                       const rocsparse_int* bsr_col_ind_A,
                       const T*             beta,
                       const rocsparse_int* bsr_row_ptr_B,
                       const rocsparse_int* bsr_col_ind_B,
                       rocsparse_int*       bsr_row_ptr_C,
                       rocsparse_int*       nnzb_C,
                       rocsparse_index_base base_A,
                       rocsparse_index_base base_B,
                       rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return host_csrgeam_nnz(Mb,
                            Nb,
                            alpha,
                            bsr_row_ptr_A,
                            bsr_col_ind_A,
                            beta,
                            bsr_row_ptr_B,
                            bsr_col_ind_B,
                            bsr_row_ptr_C,
                            nnzb_C,
                            base_A,
                            base_B,
                            base_C);
}

template <typename T>
void host_bsrgeam(rocsparse_direction  dir,
                  rocsparse_int        Mb,
                  rocsparse_int        Nb,
                  rocsparse_int        block_dim,
                  const T*             alpha,
                  const rocsparse_int* bsr_row_ptr_A,
                  const rocsparse_int* bsr_col_ind_A,
                  const T*             bsr_val_A,
                  const T*             beta,
                  const rocsparse_int* bsr_row_ptr_B,
                  const rocsparse_int* bsr_col_ind_B,
                  const T*             bsr_val_B,
                  const rocsparse_int* bsr_row_ptr_C,
                  rocsparse_int*       bsr_col_ind_C,
                  T*                   bsr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<rocsparse_int> nnzb(Nb, -1);

#ifdef _OPENMP
        rocsparse_int nthreads = omp_get_num_threads();
        rocsparse_int tid      = omp_get_thread_num();
#else
        rocsparse_int nthreads = 1;
        rocsparse_int tid      = 0;
#endif

        rocsparse_int rows_per_thread = (Mb + nthreads - 1) / nthreads;
        rocsparse_int chunk_begin     = rows_per_thread * tid;
        rocsparse_int chunk_end       = std::min(chunk_begin + rows_per_thread, Mb);

        // Loop over rows
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            rocsparse_int row_begin_C = bsr_row_ptr_C[i] - base_C;
            rocsparse_int row_end_C   = row_begin_C;

            rocsparse_int row_begin_A = bsr_row_ptr_A[i] - base_A;
            rocsparse_int row_end_A   = bsr_row_ptr_A[i + 1] - base_A;

            // Copy A into C
            for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                rocsparse_int col_A = bsr_col_ind_A[j] - base_A;

                nnzb[col_A] = row_end_C;

                bsr_col_ind_C[row_end_C] = col_A + base_C;

                for(rocsparse_int r = 0; r < block_dim; r++)
                {
                    for(rocsparse_int c = 0; c < block_dim; c++)
                    {
                        if(dir == rocsparse_direction_row)
                        {
                            bsr_val_C[block_dim * block_dim * row_end_C + block_dim * r + c]
                                = *alpha * bsr_val_A[block_dim * block_dim * j + block_dim * r + c];
                        }
                        else
                        {
                            bsr_val_C[block_dim * block_dim * row_end_C + block_dim * c + r]
                                = *alpha * bsr_val_A[block_dim * block_dim * j + block_dim * c + r];
                        }
                    }
                }

                ++row_end_C;
            }

            rocsparse_int row_begin_B = bsr_row_ptr_B[i] - base_B;
            rocsparse_int row_end_B   = bsr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(rocsparse_int j = row_begin_B; j < row_end_B; ++j)
            {
                // Current column of B
                rocsparse_int col_B = bsr_col_ind_B[j] - base_B;

                // Check if a new nnz is generated or if the value is added
                if(nnzb[col_B] < row_begin_C)
                {
                    nnzb[col_B] = row_end_C;

                    bsr_col_ind_C[row_end_C] = col_B + base_C;

                    for(rocsparse_int r = 0; r < block_dim; r++)
                    {
                        for(rocsparse_int c = 0; c < block_dim; c++)
                        {
                            if(dir == rocsparse_direction_row)
                            {
                                bsr_val_C[block_dim * block_dim * row_end_C + block_dim * r + c]
                                    = *beta
                                      * bsr_val_B[block_dim * block_dim * j + block_dim * r + c];
                            }
                            else
                            {
                                bsr_val_C[block_dim * block_dim * row_end_C + block_dim * c + r]
                                    = *beta
                                      * bsr_val_B[block_dim * block_dim * j + block_dim * c + r];
                            }
                        }
                    }

                    ++row_end_C;
                }
                else
                {
                    for(rocsparse_int r = 0; r < block_dim; r++)
                    {
                        for(rocsparse_int c = 0; c < block_dim; c++)
                        {
                            if(dir == rocsparse_direction_row)
                            {
                                bsr_val_C[block_dim * block_dim * nnzb[col_B] + block_dim * r + c]
                                    = std::fma(
                                        *beta,
                                        bsr_val_B[block_dim * block_dim * j + block_dim * r + c],
                                        bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                  + block_dim * r + c]);
                            }
                            else
                            {
                                bsr_val_C[block_dim * block_dim * nnzb[col_B] + block_dim * c + r]
                                    = std::fma(
                                        *beta,
                                        bsr_val_B[block_dim * block_dim * j + block_dim * c + r],
                                        bsr_val_C[block_dim * block_dim * nnzb[col_B]
                                                  + block_dim * c + r]);
                            }
                        }
                    }
                }
            }
        }
    }

    rocsparse_int nnzb = bsr_row_ptr_C[Mb] - base_C;

    std::vector<rocsparse_int> col(nnzb);
    std::vector<T>             val(block_dim * block_dim * nnzb);

    std::copy(bsr_col_ind_C, bsr_col_ind_C + nnzb, col.begin());
    std::copy(bsr_val_C, bsr_val_C + block_dim * block_dim * nnzb, val.begin());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < Mb; ++i)
    {
        rocsparse_int row_begin = bsr_row_ptr_C[i] - base_C;
        rocsparse_int row_end   = bsr_row_ptr_C[i + 1] - base_C;
        rocsparse_int row_nnzb  = row_end - row_begin;

        std::vector<rocsparse_int> perm(row_nnzb);
        for(rocsparse_int j = 0; j < row_nnzb; ++j)
        {
            perm[j] = j;
        }

        rocsparse_int* col_entry = col.data() + row_begin;
        T*             val_entry = val.data() + block_dim * block_dim * row_begin;

        std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
            return col_entry[a] < col_entry[b];
        });

        for(rocsparse_int j = 0; j < row_nnzb; ++j)
        {
            bsr_col_ind_C[row_begin + j] = col_entry[perm[j]];

            for(rocsparse_int r = 0; r < block_dim; r++)
            {
                for(rocsparse_int c = 0; c < block_dim; c++)
                {
                    if(dir == rocsparse_direction_row)
                    {
                        bsr_val_C[block_dim * block_dim * (row_begin + j) + block_dim * r + c]
                            = val_entry[block_dim * block_dim * perm[j] + block_dim * r + c];
                    }
                    else
                    {
                        bsr_val_C[block_dim * block_dim * (row_begin + j) + block_dim * c + r]
                            = val_entry[block_dim * block_dim * perm[j] + block_dim * c + r];
                    }
                }
            }
        }
    }
}

template <typename T, typename I, typename J>
void host_csrgeam_nnz(J                    M,
                      J                    N,
                      const T*             alpha,
                      const I*             csr_row_ptr_A,
                      const J*             csr_col_ind_A,
                      const T*             beta,
                      const I*             csr_row_ptr_B,
                      const J*             csr_col_ind_B,
                      I*                   csr_row_ptr_C,
                      I*                   nnz_C,
                      rocsparse_index_base base_A,
                      rocsparse_index_base base_B,
                      rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(M == 0 || N == 0)
    {
        *nnz_C = 0;
        for(J i = 0; i <= M; ++i)
        {
            csr_row_ptr_C[i] = base_C;
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<J> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int           nthreads = 1;
        int           tid      = 0;
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            I row_begin_A = csr_row_ptr_A[i] - base_A;
            I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Loop over columns of A
            for(I j = row_begin_A; j < row_end_A; ++j)
            {
                J col_A = csr_col_ind_A[j] - base_A;

                nnz[col_A] = i;
                ++csr_row_ptr_C[i + 1];
            }

            I row_begin_B = csr_row_ptr_B[i] - base_B;
            I row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(I j = row_begin_B; j < row_end_B; ++j)
            {
                J col_B = csr_col_ind_B[j] - base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(J i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[M] - base_C;
}

template <typename T, typename I, typename J>
void host_csrgeam(J                    M,
                  J                    N,
                  const T*             alpha,
                  const I*             csr_row_ptr_A,
                  const J*             csr_col_ind_A,
                  const T*             csr_val_A,
                  const T*             beta,
                  const I*             csr_row_ptr_B,
                  const J*             csr_col_ind_B,
                  const T*             csr_val_B,
                  const I*             csr_row_ptr_C,
                  J*                   csr_col_ind_C,
                  T*                   csr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    T alpha_val = (alpha == nullptr) ? 0 : *alpha;
    T beta_val  = (beta == nullptr) ? 0 : *beta;

    if(M == 0 || N == 0)
    {
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<J> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int           nthreads = 1;
        int           tid      = 0;
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            I row_begin_C = csr_row_ptr_C[i] - base_C;
            I row_end_C   = row_begin_C;

            I row_begin_A = csr_row_ptr_A[i] - base_A;
            I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Copy A into C
            for(I j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                J col_A = csr_col_ind_A[j] - base_A;

                // Current value of A
                T val_A = alpha_val * csr_val_A[j];

                nnz[col_A] = row_end_C;

                csr_col_ind_C[row_end_C] = col_A + base_C;
                csr_val_C[row_end_C]     = val_A;
                ++row_end_C;
            }

            I row_begin_B = csr_row_ptr_B[i] - base_B;
            I row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(I j = row_begin_B; j < row_end_B; ++j)
            {
                // Current column of B
                J col_B = csr_col_ind_B[j] - base_B;

                // Current value of B
                T val_B = beta_val * csr_val_B[j];

                // Check if a new nnz is generated or if the value is added
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_B + base_C;
                    csr_val_C[row_end_C]     = val_B;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] += val_B;
                }
            }
        }
    }

    I nnz = csr_row_ptr_C[M] - base_C;

    std::vector<J> col(nnz);
    std::vector<T> val(nnz);

    std::copy(csr_col_ind_C, csr_col_ind_C + nnz, col.begin());
    std::copy(csr_val_C, csr_val_C + nnz, val.begin());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr_C[i] - base_C;
        I row_end   = csr_row_ptr_C[i + 1] - base_C;
        J row_nnz   = row_end - row_begin;

        std::vector<J> perm(row_nnz);
        for(J j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        J* col_entry = col.data() + row_begin;
        T* val_entry = val.data() + row_begin;

        std::sort(perm.begin(), perm.end(), [&](const J& a, const J& b) {
            return col_entry[a] < col_entry[b];
        });

        for(J j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

template <typename T, typename I, typename J>
void host_csrgemm_nnz(J                    M,
                      J                    N,
                      J                    K,
                      const T*             alpha,
                      const I*             csr_row_ptr_A,
                      const J*             csr_col_ind_A,
                      const I*             csr_row_ptr_B,
                      const J*             csr_col_ind_B,
                      const T*             beta,
                      const I*             csr_row_ptr_D,
                      const J*             csr_col_ind_D,
                      I*                   csr_row_ptr_C,
                      I*                   nnz_C,
                      rocsparse_index_base base_A,
                      rocsparse_index_base base_B,
                      rocsparse_index_base base_C,
                      rocsparse_index_base base_D)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(M == 0 || N == 0)
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }
    else if(alpha && !beta && (K == 0))
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }
    else if(!alpha && !beta)
    {
        *nnz_C = 0;
        if(M > 0)
        {
            for(J i = 0; i <= M; ++i)
            {
                csr_row_ptr_C[i] = base_C;
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<J> nnz(N, -1);

        int nthreads = 1;
        int tid      = 0;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid      = omp_get_thread_num();
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - base_A;

                    I row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(I k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[k] - base_B;

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            ++csr_row_ptr_C[i + 1];
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                I row_begin_D = csr_row_ptr_D[i] - base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    J col_D = csr_col_ind_D[j] - base_D;

                    // Check if a new nnz is generated
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        ++csr_row_ptr_C[i + 1];
                    }
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(J i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[M] - base_C;
}

template <typename T, typename I, typename J>
void host_csrgemm(J                    M,
                  J                    N,
                  J                    L,
                  const T*             alpha,
                  const I*             csr_row_ptr_A,
                  const J*             csr_col_ind_A,
                  const T*             csr_val_A,
                  const I*             csr_row_ptr_B,
                  const J*             csr_col_ind_B,
                  const T*             csr_val_B,
                  const T*             beta,
                  const I*             csr_row_ptr_D,
                  const J*             csr_col_ind_D,
                  const T*             csr_val_D,
                  const I*             csr_row_ptr_C,
                  J*                   csr_col_ind_C,
                  T*                   csr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C,
                  rocsparse_index_base base_D)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(M == 0 || N == 0)
    {
        return;
    }
    else if(alpha && !beta && (L == 0))
    {
        return;
    }
    else if(!alpha && !beta)
    {
        return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<I> nnz(N, -1);

        int nthreads = 1;
        int tid      = 0;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid      = omp_get_thread_num();
#endif

        J rows_per_thread = (M + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            I row_begin_C = csr_row_ptr_C[i] - base_C;
            I row_end_C   = row_begin_C;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - base_A;
                    // Current value of A
                    T val_A = *alpha * csr_val_A[j];

                    I row_begin_B = csr_row_ptr_B[col_A] - base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - base_B;

                    // Loop over columns of B in row col_A
                    for(I k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[k] - base_B;
                        // Current value of B
                        T val_B = csr_val_B[k];

                        // Check if a new nnz is generated or if the product is appended
                        if(nnz[col_B] < row_begin_C)
                        {
                            nnz[col_B]               = row_end_C;
                            csr_col_ind_C[row_end_C] = col_B + base_C;
                            csr_val_C[row_end_C]     = val_A * val_B;
                            ++row_end_C;
                        }
                        else
                        {
                            csr_val_C[nnz[col_B]] += val_A * val_B;
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                I row_begin_D = csr_row_ptr_D[i] - base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    J col_D = csr_col_ind_D[j] - base_D;
                    // Current value of D
                    T val_D = *beta * csr_val_D[j];

                    // Check if a new nnz is generated or if the value is added
                    if(nnz[col_D] < row_begin_C)
                    {
                        nnz[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D + base_C;
                        csr_val_C[row_end_C]     = val_D;
                        ++row_end_C;
                    }
                    else
                    {
                        csr_val_C[nnz[col_D]] += val_D;
                    }
                }
            }
        }
    }

    I nnz = csr_row_ptr_C[M] - base_C;

    std::vector<J> col(nnz);
    std::vector<T> val(nnz);

    memcpy(col.data(), csr_col_ind_C, sizeof(J) * nnz);
    memcpy(val.data(), csr_val_C, sizeof(T) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr_C[i] - base_C;
        I row_end   = csr_row_ptr_C[i + 1] - base_C;
        J row_nnz   = row_end - row_begin;

        std::vector<J> perm(row_nnz);
        for(J j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        J* col_entry = col.data() + row_begin;
        T* val_entry = val.data() + row_begin;

        std::sort(perm.begin(), perm.end(), [&](const J& a, const J& b) {
            return col_entry[a] < col_entry[b];
        });

        for(J j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void rocsparse_host<T, I, J, A, B, C>::cooddmm(rocsparse_operation  transA,
                                               rocsparse_operation  transB,
                                               rocsparse_order      orderA,
                                               rocsparse_order      orderB,
                                               J                    M,
                                               J                    N,
                                               J                    K,
                                               I                    nnz,
                                               const T*             alpha,
                                               const A*             dense_A,
                                               int64_t              lda,
                                               const B*             dense_B,
                                               int64_t              ldb,
                                               const T*             beta,
                                               const I*             coo_row_ind_C,
                                               const I*             coo_col_ind_C,
                                               C*                   coo_val_C,
                                               rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const T a = *alpha;
    const T b = *beta;

    const int64_t incx = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? lda : 1)
                             : ((transA == rocsparse_operation_none) ? 1 : lda);
    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I s = 0; s < nnz; ++s)
    {
        const I i = coo_row_ind_C[s] - base_C;
        const I j = coo_col_ind_C[s] - base_C;

        const A* x
            = (orderA == rocsparse_order_column)
                  ? ((transA == rocsparse_operation_none) ? (dense_A + i) : (dense_A + lda * i))
                  : ((transA == rocsparse_operation_none) ? (dense_A + lda * i) : (dense_A + i));

        const B* y
            = (orderB == rocsparse_order_column)
                  ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j) : (dense_B + j))
                  : ((transB == rocsparse_operation_none) ? (dense_B + j) : (dense_B + ldb * j));

        T sum = static_cast<T>(0);
        for(J k = 0; k < K; ++k)
        {
            sum += x[incx * k] * y[incy * k];
        }
        coo_val_C[s] = b * coo_val_C[s] + a * sum;
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void rocsparse_host<T, I, J, A, B, C>::cooaosddmm(rocsparse_operation  transA,
                                                  rocsparse_operation  transB,
                                                  rocsparse_order      orderA,
                                                  rocsparse_order      orderB,
                                                  J                    M,
                                                  J                    N,
                                                  J                    K,
                                                  I                    nnz,
                                                  const T*             alpha,
                                                  const A*             dense_A,
                                                  int64_t              lda,
                                                  const B*             dense_B,
                                                  int64_t              ldb,
                                                  const T*             beta,
                                                  const I*             coo_row_ind_C,
                                                  const I*             coo_col_ind_C,
                                                  C*                   coo_val_C,
                                                  rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const T a = *alpha;
    const T b = *beta;

    const int64_t incx = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? lda : 1)
                             : ((transA == rocsparse_operation_none) ? 1 : lda);
    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(I s = 0; s < nnz; ++s)
    {
        const I i = coo_row_ind_C[2 * s] - base_C;
        const I j = coo_col_ind_C[2 * s] - base_C;

        const A* x
            = (orderA == rocsparse_order_column)
                  ? ((transA == rocsparse_operation_none) ? (dense_A + i) : (dense_A + lda * i))
                  : ((transA == rocsparse_operation_none) ? (dense_A + lda * i) : (dense_A + i));

        const B* y
            = (orderB == rocsparse_order_column)
                  ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j) : (dense_B + j))
                  : ((transB == rocsparse_operation_none) ? (dense_B + j) : (dense_B + ldb * j));

        T sum = static_cast<T>(0);
        for(J k = 0; k < K; ++k)
        {
            sum += x[incx * k] * y[incy * k];
        }
        coo_val_C[s] = b * coo_val_C[s] + a * sum;
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void rocsparse_host<T, I, J, A, B, C>::csrddmm(rocsparse_operation  transA,
                                               rocsparse_operation  transB,
                                               rocsparse_order      orderA,
                                               rocsparse_order      orderB,
                                               J                    M,
                                               J                    N,
                                               J                    K,
                                               I                    nnz,
                                               const T*             alpha,
                                               const A*             dense_A,
                                               int64_t              lda,
                                               const B*             dense_B,
                                               int64_t              ldb,
                                               const T*             beta,
                                               const I*             csr_row_ptr_C,
                                               const J*             csr_col_ind_C,
                                               C*                   csr_val_C,
                                               rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const T a = *alpha;
    const T b = *beta;

    const int64_t incx = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? lda : 1)
                             : ((transA == rocsparse_operation_none) ? 1 : lda);

    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        const I start = csr_row_ptr_C[i] - base_C;
        const I end   = csr_row_ptr_C[i + 1] - base_C;

        for(I at = start; at < end; ++at)
        {
            J j = csr_col_ind_C[at] - base_C;

            const A* x
                = (orderA == rocsparse_order_column)
                      ? ((transA == rocsparse_operation_none) ? (dense_A + i) : (dense_A + lda * i))
                      : ((transA == rocsparse_operation_none) ? (dense_A + lda * i)
                                                              : (dense_A + i));
            const B* y
                = (orderB == rocsparse_order_column)
                      ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j) : (dense_B + j))
                      : ((transB == rocsparse_operation_none) ? (dense_B + j)
                                                              : (dense_B + ldb * j));

            T sum = static_cast<T>(0);
            for(J k = 0; k < K; ++k)
            {
                sum += x[incx * k] * y[incy * k];
            }
            csr_val_C[at] = b * csr_val_C[at] + a * sum;
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void rocsparse_host<T, I, J, A, B, C>::ellddmm(rocsparse_operation  transA,
                                               rocsparse_operation  transB,
                                               rocsparse_order      orderA,
                                               rocsparse_order      orderB,
                                               J                    M,
                                               J                    N,
                                               J                    K,
                                               I                    nnz,
                                               const T*             alpha,
                                               const A*             dense_A,
                                               int64_t              lda,
                                               const B*             dense_B,
                                               int64_t              ldb,
                                               const T*             beta,
                                               const J              ell_width,
                                               const I*             ell_ind_C,
                                               C*                   ell_val_C,
                                               rocsparse_index_base ell_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const T a = *alpha;
    const T b = *beta;

    const int64_t incx = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? lda : 1)
                             : ((transA == rocsparse_operation_none) ? 1 : lda);

    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < M; ++i)
    {
        for(J p = 0; p < ell_width; ++p)
        {
            I at = p * M + i;
            J j  = ell_ind_C[at] - ell_base;
            if(j >= 0 && j < N)
            {
                const A* x = (orderA == rocsparse_order_column)
                                 ? ((transA == rocsparse_operation_none) ? (dense_A + i)
                                                                         : (dense_A + lda * i))
                                 : ((transA == rocsparse_operation_none) ? (dense_A + lda * i)
                                                                         : (dense_A + i));
                const B* y = (orderB == rocsparse_order_column)
                                 ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j)
                                                                         : (dense_B + j))
                                 : ((transB == rocsparse_operation_none) ? (dense_B + j)
                                                                         : (dense_B + ldb * j));

                T sum = static_cast<T>(0);
                for(J k = 0; k < K; ++k)
                {
                    sum += x[incx * k] * y[incy * k];
                }
                ell_val_C[at] = b * ell_val_C[at] + a * sum;
            }
        }
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
void rocsparse_host<T, I, J, A, B, C>::cscddmm(rocsparse_operation  transA,
                                               rocsparse_operation  transB,
                                               rocsparse_order      orderA,
                                               rocsparse_order      orderB,
                                               J                    M,
                                               J                    N,
                                               J                    K,
                                               I                    nnz,
                                               const T*             alpha,
                                               const A*             dense_A,
                                               int64_t              lda,
                                               const B*             dense_B,
                                               int64_t              ldb,
                                               const T*             beta,
                                               const I*             csr_ptr_C,
                                               const J*             csr_ind_C,
                                               C*                   csr_val_C,
                                               rocsparse_index_base base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const T a = *alpha;
    const T b = *beta;

    const int64_t incx = (orderA == rocsparse_order_column)
                             ? ((transA == rocsparse_operation_none) ? lda : 1)
                             : ((transA == rocsparse_operation_none) ? 1 : lda);

    const int64_t incy = (orderB == rocsparse_order_column)
                             ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                             : ((transB == rocsparse_operation_none) ? ldb : 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif

    for(J j = 0; j < N; ++j)
    {
        for(I at = csr_ptr_C[j] - base_C; at < csr_ptr_C[j + 1] - base_C; ++at)
        {
            J        i = csr_ind_C[at] - base_C;
            const A* x
                = (orderA == rocsparse_order_column)
                      ? ((transA == rocsparse_operation_none) ? (dense_A + i) : (dense_A + lda * i))
                      : ((transA == rocsparse_operation_none) ? (dense_A + lda * i)
                                                              : (dense_A + i));
            const B* y
                = (orderB == rocsparse_order_column)
                      ? ((transB == rocsparse_operation_none) ? (dense_B + ldb * j) : (dense_B + j))
                      : ((transB == rocsparse_operation_none) ? (dense_B + j)
                                                              : (dense_B + ldb * j));

            T sum = static_cast<T>(0);
            for(J k = 0; k < K; ++k)
            {
                sum += x[incx * k] * y[incy * k];
            }
            csr_val_C[at] = b * csr_val_C[at] + a * sum;
        }
    }
}

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsric0(rocsparse_direction               direction,
                 rocsparse_int                     Mb,
                 rocsparse_int                     block_dim,
                 const std::vector<rocsparse_int>& bsr_row_ptr,
                 const std::vector<rocsparse_int>& bsr_col_ind,
                 std::vector<T>&                   bsr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot)

{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int M = Mb * block_dim;

    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_block_offset(Mb);
    std::vector<rocsparse_int> diag_offset(M, -1);
    std::vector<rocsparse_int> nnz_entries(M, -1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < Mb; i++)
    {
        rocsparse_int row_begin = bsr_row_ptr[i] - base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            if(bsr_col_ind[j] - base == i)
            {
                diag_block_offset[i] = j;
                break;
            }
        }
    }

    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int local_row = i % block_dim;

        rocsparse_int row_begin = bsr_row_ptr[i / block_dim] - base;
        rocsparse_int row_end   = bsr_row_ptr[i / block_dim + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * local_row + k;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * k + local_row;
                }
            }
        }

        T             sum            = static_cast<T>(0);
        rocsparse_int diag_val_index = -1;

        bool has_diag         = false;
        bool break_outer_loop = false;

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                rocsparse_int col_j = block_dim * block_col_j + k;

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    diag_val_index = block_dim * block_dim * j + block_dim * k + k;

                    has_diag         = true;
                    break_outer_loop = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break_outer_loop = true;
                    break;
                }

                T val_j = static_cast<T>(0);
                if(direction == rocsparse_direction_row)
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                }
                else
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                }

                rocsparse_int local_row_j = col_j % block_dim;

                rocsparse_int row_begin_j = bsr_row_ptr[col_j / block_dim] - base;
                rocsparse_int row_end_j   = diag_block_offset[col_j / block_dim];
                rocsparse_int row_diag_j  = diag_offset[col_j];

                T local_sum = static_cast<T>(0);
                T inv_diag  = row_diag_j != -1 ? bsr_val[row_diag_j] : static_cast<T>(0);

                // Check for numeric zero
                if(inv_diag == static_cast<T>(0))
                {
                    // Numerical non-invertible block diagonal
                    if(*numeric_pivot == -1)
                    {
                        *numeric_pivot = block_col_j + base;
                    }

                    *numeric_pivot = std::min(*numeric_pivot, block_col_j + base);

                    inv_diag = static_cast<T>(1);
                }

                inv_diag = static_cast<T>(1) / inv_diag;

                // loop over upper offset pointer and do linear combination for nnz entry
                for(rocsparse_int l = row_begin_j; l < row_end_j + 1; l++)
                {
                    rocsparse_int block_col_l = bsr_col_ind[l] - base;

                    for(rocsparse_int m = 0; m < block_dim; m++)
                    {
                        rocsparse_int idx = nnz_entries[block_dim * block_col_l + m];

                        if(idx != -1 && block_dim * block_col_l + m < col_j)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * local_row_j + m],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                            else
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * m + local_row_j],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                        }
                    }
                }

                val_j = (val_j - local_sum) * inv_diag;
                sum   = std::fma(val_j, rocsparse_conj(val_j), sum);

                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val_j;
                }
                else
                {
                    bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val_j;
                }
            }

            if(break_outer_loop)
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural missing block diagonal
            if(*struct_pivot == -1)
            {
                *struct_pivot = i / block_dim + base;
            }
        }

        // Process diagonal entry
        if(has_diag)
        {
            T diag_entry            = std::sqrt(std::abs(bsr_val[diag_val_index] - sum));
            bsr_val[diag_val_index] = diag_entry;

            if(diag_entry == static_cast<T>(0))
            {
                // Numerical non-invertible block diagonal
                if(*numeric_pivot == -1)
                {
                    *numeric_pivot = i / block_dim + base;
                }

                *numeric_pivot = std::min(*numeric_pivot, i / block_dim + base);
            }

            // Store diagonal offset
            diag_offset[i] = diag_val_index;
        }

        for(rocsparse_int j = row_begin; j < row_end; j++)
        {
            rocsparse_int block_col_j = bsr_col_ind[j] - base;

            for(rocsparse_int k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
            }
        }
    }
}

template <typename T, typename U>
void host_bsrilu0(rocsparse_direction               dir,
                  rocsparse_int                     mb,
                  const std::vector<rocsparse_int>& bsr_row_ptr,
                  const std::vector<rocsparse_int>& bsr_col_ind,
                  std::vector<T>&                   bsr_val,
                  rocsparse_int                     bsr_dim,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val)

{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivots
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    // Temporary vector to hold diagonal offset to access diagonal BSR block
    std::vector<rocsparse_int> diag_offset(mb);
    std::vector<rocsparse_int> nnz_entries(mb, -1);

    if(mb > 0)
    {
        // First diagonal block is index 0
        diag_offset[0] = 0;
    }

    // Loop over all BSR rows
    for(rocsparse_int i = 0; i < mb; ++i)
    {
        // Flag whether we have a diagonal block or not
        bool has_diag = false;

        // BSR column entry and exit point
        rocsparse_int row_begin = bsr_row_ptr[i] - base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        rocsparse_int j;

        // Set up entry points for linear combination
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = bsr_col_ind[j] - base;
            nnz_entries[col_j]  = j;
        }

        // Process lower diagonal BSR blocks (diagonal BSR block is excluded)
        for(j = row_begin; j < row_end; ++j)
        {
            // Column index of current BSR block
            rocsparse_int bsr_col = bsr_col_ind[j] - base;

            // If this is a diagonal block, set diagonal flag to true and skip
            // all upcoming blocks as we exceed the lower matrix part
            if(bsr_col == i)
            {
                has_diag = true;
                break;
            }

            // Skip all upper matrix blocks
            if(bsr_col > i)
            {
                break;
            }

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            rocsparse_int diag_j    = diag_offset[bsr_col];
            rocsparse_int row_end_j = bsr_row_ptr[bsr_col + 1] - base;

            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                // Process all rows within the BSR block
                for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through columns of bk-th row and do linear combination
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = diag_j + 1; k < row_end_j; ++k)
            {
                rocsparse_int bsr_col_k = bsr_col_ind[k] - base;

                if(nnz_entries[bsr_col_k] != -1)
                {
                    rocsparse_int m = nnz_entries[bsr_col_k];

                    // Loop through all rows within the BSR block
                    for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
                    {
                        // Loop through columns of bi-th row and do linear combination
                        for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                        {
                            T sum = static_cast<T>(0);

                            for(rocsparse_int bk = 0; bk < bsr_dim; ++bk)
                            {
                                sum = std::fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                               bsr_val[BSR_IND(k, bk, bj, dir)],
                                               sum);
                            }

                            bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                        }
                    }
                }
            }
        }

        // Check for structural pivot
        if(!has_diag)
        {
            *struct_pivot = std::min(*struct_pivot, i + base);
            break;
        }

        // Process diagonal
        if(bsr_col_ind[j] - base == i)
        {
            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(j, bi, bi, dir)];

                if(boost)
                {
                    diag = (boost_tol >= std::abs(diag)) ? boost_val : diag;
                    bsr_val[BSR_IND(j, bi, bi, dir)] = diag;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == static_cast<T>(0))
                    {
                        *numeric_pivot = std::min(*numeric_pivot, bsr_col_ind[j]);
                        continue;
                    }
                }

                // Process all rows within the BSR block after bi-th row
                for(rocsparse_int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through remaining columns of bk-th row and do linear combination
                    for(rocsparse_int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Store diagonal BSR block entry point
        rocsparse_int row_diag = diag_offset[i] = j;

        // Process upper diagonal BSR blocks
        for(j = row_diag + 1; j < row_end; ++j)
        {
            // Loop through all rows within the BSR block
            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // Process all rows within the BSR block after bi-th row
                for(rocsparse_int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    // Loop through columns of bk-th row and do linear combination
                    for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-bsr_val[BSR_IND(row_diag, bk, bi, dir)],
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Reset entry points
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = bsr_col_ind[j] - base;
            nnz_entries[col_j]  = -1;
        }
    }

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename T>
void host_csric0(rocsparse_int                     M,
                 const std::vector<rocsparse_int>& csr_row_ptr,
                 const std::vector<rocsparse_int>& csr_col_ind,
                 std::vector<T>&                   csr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot,
                 rocsparse_int*                    singular_pivot,
                 double                            tol)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot   = -1;
    *numeric_pivot  = -1;
    *singular_pivot = -1;

    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_offset(M);
    std::vector<rocsparse_int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(rocsparse_int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        rocsparse_int row_begin = csr_row_ptr[ai] - base;
        rocsparse_int row_end   = csr_row_ptr[ai + 1] - base;
        rocsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        T sum = static_cast<T>(0);

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col_j = csr_col_ind[j] - base;
            T             val_j = csr_val[j];

            // Mark diagonal and skip row
            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            // Skip upper triangular
            if(col_j > ai)
            {
                break;
            }

            rocsparse_int row_begin_j = csr_row_ptr[col_j] - base;
            rocsparse_int row_diag_j  = diag_offset[col_j];

            T local_sum = static_cast<T>(0);
            T diag_val  = csr_val[row_diag_j];
            T inv_diag  = static_cast<T>(0);

            // Check for numeric negative
            if((std::real(diag_val) <= tol) && (std::imag(diag_val) == 0))
            {
                // Numerical negative diagonal
                *singular_pivot = (*singular_pivot == -1) ? (col_j + base)
                                                          : std::min(*singular_pivot, col_j + base);
            }

            // Check for numeric zero
            if(diag_val == static_cast<T>(0))
            {
                // Numerical zero diagonal
                *numeric_pivot = (*numeric_pivot == -1) ? (col_j + base)
                                                        : std::min(*numeric_pivot, col_j + base);
            }
            else
            {

                inv_diag = static_cast<T>(1) / diag_val;
            }

            // loop over upper offset pointer and do linear combination for nnz entry
            for(rocsparse_int k = row_begin_j; k < row_diag_j; ++k)
            {
                rocsparse_int col_k = csr_col_ind[k] - base;

                // if nnz at this position do linear combination
                if(nnz_entries[col_k] != 0)
                {
                    rocsparse_int idx = nnz_entries[col_k];
                    local_sum = std::fma(csr_val[k], rocsparse_conj(csr_val[idx]), local_sum);
                }
            }

            val_j = (val_j - local_sum) * inv_diag;
            sum   = std::fma(val_j, rocsparse_conj(val_j), sum);

            csr_val[j] = val_j;
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot
                = (*struct_pivot == -1) ? (ai + base) : std::min(*struct_pivot, ai + base);
            *numeric_pivot
                = (*numeric_pivot == -1) ? (ai + base) : std::min(*numeric_pivot, ai + base);
        }
        else
        {
            // Store diagonal offset
            diag_offset[ai] = j;

            // Process diagonal entry
            T diag_entry = csr_val[j] - sum;
            csr_val[j]   = std::sqrt(std::abs(diag_entry));

            auto tolXtol = tol * tol;
            if((std::real(diag_entry) <= tolXtol) && (std::imag(diag_entry) == 0))
            {
                *singular_pivot
                    = (*singular_pivot == -1) ? (ai + base) : std::min(*singular_pivot, ai + base);
            }

            // check for zero diagonal
            if(diag_entry == static_cast<T>(0))
            {
                *numeric_pivot
                    = (*numeric_pivot == -1) ? (ai + base) : std::min(*numeric_pivot, ai + base);
            }
        }

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
        }
    }

    if(*struct_pivot != -1)
    {
        *numeric_pivot
            = (*numeric_pivot == -1) ? (*struct_pivot) : std::min(*numeric_pivot, *struct_pivot);
    }

    if(*numeric_pivot != -1)
    {
        *singular_pivot = (*singular_pivot == -1) ? (*numeric_pivot)
                                                  : std::min(*singular_pivot, *numeric_pivot);
    }
}

template <typename T, typename U>
void host_csrilu0(rocsparse_int                     M,
                  const std::vector<rocsparse_int>& csr_row_ptr,
                  const std::vector<rocsparse_int>& csr_col_ind,
                  std::vector<T>&                   csr_val,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  rocsparse_int*                    singular_pivot,
                  double                            tol,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if((struct_pivot == nullptr) || (numeric_pivot == nullptr) || (singular_pivot == nullptr))
    {
        std::cerr << "host_csrilu0: invalid parameters, failed condition: ((struct_pivot != "
                     "nullptr) && (numeric_pivot != nullptr) && (singular_pivot != nullptr))."
                  << std::endl;
        throw(1);
    }

    // Initialize pivot
    *struct_pivot   = -1;
    *numeric_pivot  = -1;
    *singular_pivot = -1;

    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_offset(M, -1);
    std::vector<rocsparse_int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(rocsparse_int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        rocsparse_int row_begin = csr_row_ptr[ai] - base;
        rocsparse_int row_end   = csr_row_ptr[ai + 1] - base;
        rocsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        bool          has_diag = false;
        rocsparse_int diag_pos = -1;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(csr_col_ind[j] - base < ai)
            {

                rocsparse_int col_j  = csr_col_ind[j] - base;
                rocsparse_int diag_j = diag_offset[col_j];
                if(diag_j < 0)
                    continue;

                T diag_val = csr_val[diag_j];

                if(boost)
                {
                    diag_val        = (boost_tol >= std::abs(diag_val)) ? boost_val : diag_val;
                    csr_val[diag_j] = diag_val;
                }
                else
                {

                    // Check for numeric singular pivot
                    if(std::abs(diag_val) <= tol)
                    {
                        *singular_pivot = (*singular_pivot == -1)
                                              ? (col_j + base)
                                              : std::min(*singular_pivot, col_j + base);
                    }

                    // Check for numeric zero pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        *numeric_pivot = (*numeric_pivot == -1)
                                             ? (col_j + base)
                                             : std::min(*numeric_pivot, col_j + base);
                        continue;
                    }
                }

                {
                    // multiplication factor

                    csr_val[j] = csr_val[j] / diag_val;

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(rocsparse_int k = diag_j + 1; k < csr_row_ptr[col_j + 1] - base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[csr_col_ind[k] - base] != 0)
                        {
                            rocsparse_int idx = nnz_entries[csr_col_ind[k] - base];
                            csr_val[idx]      = std::fma(-csr_val[j], csr_val[k], csr_val[idx]);
                        }
                    }
                }
            }
            else if(csr_col_ind[j] - base == ai)
            {
                has_diag = true;
                diag_pos = j;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot
                = (*struct_pivot == -1) ? (ai + base) : std::min(*struct_pivot, ai + base);
            *numeric_pivot
                = (*numeric_pivot == -1) ? (ai + base) : std::min(*numeric_pivot, ai + base);
        }
        else
        {
            // set diagonal pointer to diagonal element
            diag_offset[ai] = diag_pos;
            if(boost)
            {
                if(std::abs(csr_val[diag_pos]) <= boost_tol)
                {
                    csr_val[diag_pos] = boost_val;
                }
            }
            else
            {
                const rocsparse_int diag_pos = diag_offset[ai];
                const bool is_diag = (diag_pos >= 0) && (csr_col_ind[diag_pos] == (ai + base));

                const bool is_singular_diag = is_diag && (std::abs(csr_val[diag_pos]) <= tol);
                const bool is_zero_diag     = is_diag && (csr_val[diag_pos] == static_cast<T>(0));

                // check for singular diagonal
                if(is_singular_diag)
                {
                    *singular_pivot = (*singular_pivot == -1)
                                          ? (ai + base)
                                          : std::min(*singular_pivot, (ai + base));
                }

                // check for zero diagonal
                if(is_zero_diag)
                {
                    *numeric_pivot = (*numeric_pivot == -1) ? (ai + base)
                                                            : std::min(*numeric_pivot, (ai + base));
                }
            }

            // clear nnz entries
            for(j = row_begin; j < row_end; ++j)
            {
                nnz_entries[csr_col_ind[j] - base] = 0;
            }
        }
    }

    if(*struct_pivot != -1)
    {
        *numeric_pivot
            = (*numeric_pivot == -1) ? (*struct_pivot) : std::min(*numeric_pivot, *struct_pivot);
    }

    if(*numeric_pivot != -1)
    {
        *singular_pivot = (*singular_pivot == -1) ? (*numeric_pivot)
                                                  : std::min(*singular_pivot, *numeric_pivot);
    }
}

// Parallel Cyclic reduction based on paper "Fast Tridiagonal Solvers on the GPU" by Yao Zhang
template <typename T>
void host_gtsv_no_pivot(rocsparse_int         m,
                        rocsparse_int         n,
                        const std::vector<T>& dl,
                        const std::vector<T>& d,
                        const std::vector<T>& du,
                        std::vector<T>&       B,
                        rocsparse_int         ldb)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    //
    // Compute BLOCKSIZE as the lowest power of 2 greater or equal than m,
    // and compute the exponent 'iter' of (BLOCKSIZE / 2).
    // Note: if m = 1 then the calculation below leads to iter = -1 which is, in this algorithm, as acceptable as log2(m / 2) = -Inf.
    //
    rocsparse_int iter;
    size_t        BLOCKSIZE;
    for(iter = 0, BLOCKSIZE = 1; BLOCKSIZE < m; BLOCKSIZE <<= 1, ++iter)
        ;
    --iter;

    for(rocsparse_int col = 0; col < n; col++)
    {
        rocsparse_int stride = 1;

        std::vector<T> sa(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sb(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sc(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> srhs(BLOCKSIZE, static_cast<T>(0));

        std::vector<T> a(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> b(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> c(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> rhs(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> x(BLOCKSIZE, static_cast<T>(0));

        for(rocsparse_int i = 0; i < m; i++)
        {
            a[i]   = dl[i];
            b[i]   = d[i];
            c[i]   = du[i];
            rhs[i] = B[ldb * col + i];
        }

        for(rocsparse_int j = 0; j < iter; j++)
        {
            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                rocsparse_int right = tid + stride;
                if(right >= m)
                    right = m - 1;

                rocsparse_int left = tid - stride;
                if(left < 0)
                    left = 0;

                T k1 = a[tid] / b[left];
                T k2 = c[tid] / b[right];

                T tb   = b[tid] - c[left] * k1 - a[right] * k2;
                T trhs = rhs[tid] - rhs[left] * k1 - rhs[right] * k2;
                T ta   = -a[left] * k1;
                T tc   = -c[right] * k2;

                sb[tid]   = tb;
                srhs[tid] = trhs;
                sa[tid]   = ta;
                sc[tid]   = tc;
            }

            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                a[tid]   = sa[tid];
                b[tid]   = sb[tid];
                c[tid]   = sc[tid];
                rhs[tid] = srhs[tid];
            }

            stride *= 2;
        }

        for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
        {
            if(tid < BLOCKSIZE / 2)
            {
                rocsparse_int i = tid;
                rocsparse_int j = tid + stride;

                if(j < m)
                {
                    // Solve 2x2 systems
                    T det = b[j] * b[i] - c[i] * a[j];
                    x[i]  = (b[j] * rhs[i] - c[i] * rhs[j]) / det;
                    x[j]  = (rhs[j] * b[i] - rhs[i] * a[j]) / det;
                }
                else
                {
                    // Solve 1x1 systems
                    x[i] = rhs[i] / b[i];
                }
            }
        }

        for(rocsparse_int i = 0; i < m; i++)
        {
            B[ldb * col + i] = x[i];
        }
    }
}

// Parallel Cyclic reduction based on paper "Fast Tridiagonal Solvers on the GPU" by Yao Zhang
template <typename T>
void host_gtsv_no_pivot_strided_batch(rocsparse_int         m,
                                      const std::vector<T>& dl,
                                      const std::vector<T>& d,
                                      const std::vector<T>& du,
                                      std::vector<T>&       x,
                                      rocsparse_int         batch_count,
                                      rocsparse_int         batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    //
    // Compute BLOCKSIZE as the lowest power of 2 greater or equal than m,
    // and compute the exponent 'iter' of (BLOCKSIZE / 2).
    // Note: if m = 1 then the calculation below leads to iter = -1 which is, in this algorithm, as acceptable as log2(m / 2) = -Inf.
    //
    rocsparse_int iter;
    size_t        BLOCKSIZE;
    for(iter = 0, BLOCKSIZE = 1; BLOCKSIZE < m; BLOCKSIZE <<= 1, ++iter)
        ;
    --iter;

    for(rocsparse_int col = 0; col < batch_count; col++)
    {
        rocsparse_int stride = 1;

        std::vector<T> sa(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sb(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> sc(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> srhs(BLOCKSIZE, static_cast<T>(0));

        std::vector<T> a(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> b(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> c(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> rhs(BLOCKSIZE, static_cast<T>(0));
        std::vector<T> y(BLOCKSIZE, static_cast<T>(0));

        for(rocsparse_int i = 0; i < m; i++)
        {
            a[i]   = dl[batch_stride * col + i];
            b[i]   = d[batch_stride * col + i];
            c[i]   = du[batch_stride * col + i];
            rhs[i] = x[batch_stride * col + i];
        }

        for(rocsparse_int j = 0; j < iter; j++)
        {
            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                rocsparse_int right = tid + stride;
                if(right >= m)
                    right = m - 1;

                rocsparse_int left = tid - stride;
                if(left < 0)
                    left = 0;

                T k1 = a[tid] / b[left];
                T k2 = c[tid] / b[right];

                T tb   = b[tid] - c[left] * k1 - a[right] * k2;
                T trhs = rhs[tid] - rhs[left] * k1 - rhs[right] * k2;
                T ta   = -a[left] * k1;
                T tc   = -c[right] * k2;

                sb[tid]   = tb;
                srhs[tid] = trhs;
                sa[tid]   = ta;
                sc[tid]   = tc;
            }

            for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
            {
                a[tid]   = sa[tid];
                b[tid]   = sb[tid];
                c[tid]   = sc[tid];
                rhs[tid] = srhs[tid];
            }

            stride *= 2;
        }

        for(rocsparse_int tid = 0; tid < BLOCKSIZE; tid++)
        {
            if(tid < BLOCKSIZE / 2)
            {
                rocsparse_int i = tid;
                rocsparse_int j = tid + stride;

                if(j < m)
                {
                    // Solve 2x2 systems
                    T det = b[j] * b[i] - c[i] * a[j];
                    y[i]  = (b[j] * rhs[i] - c[i] * rhs[j]) / det;
                    y[j]  = (rhs[j] * b[i] - rhs[i] * a[j]) / det;
                }
                else
                {
                    // Solve 1x1 systems
                    y[i] = rhs[i] / b[i];
                }
            }
        }

        for(rocsparse_int i = 0; i < m; i++)
        {
            x[batch_stride * col + i] = y[i];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_thomas(rocsparse_int m,
                                        const T*      dl,
                                        const T*      d,
                                        const T*      du,
                                        T*            x,
                                        rocsparse_int batch_count,
                                        rocsparse_int batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    std::vector<T> c1(m * batch_count, 0);
    std::vector<T> x1(m * batch_count, 0);

    // Forward elimination
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        c1[j] = du[j] / d[j];
        x1[j] = x[j] / d[j];
    }

    for(rocsparse_int i = 1; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            rocsparse_int index = batch_count * i + j;
            rocsparse_int minus = batch_count * (i - 1) + j;

            T tdu = du[batch_stride * i + j];
            T td  = d[batch_stride * i + j];
            T tdl = dl[batch_stride * i + j];
            T tx  = x[batch_stride * i + j];

            c1[index] = tdu / (td - c1[minus] * tdl);
            x1[index] = (tx - x1[minus] * tdl) / (td - c1[minus] * tdl);
        }
    }

    // backward substitution
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j] = x1[batch_count * (m - 1) + j];
    }

    for(rocsparse_int i = m - 2; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            rocsparse_int index = batch_count * i + j;

            x[batch_stride * i + j] = x1[index] - c1[index] * x[batch_stride * (i + 1) + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_lu(rocsparse_int m,
                                    const T*      dl,
                                    const T*      d,
                                    const T*      du,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    std::vector<T>             l(m * batch_count, 0);
    std::vector<T>             u0(m * batch_count, 0);
    std::vector<T>             u1(m * batch_count, 0);
    std::vector<T>             u2(m * batch_count, 0);
    std::vector<rocsparse_int> p(m * batch_count, 0);

    for(rocsparse_int i = 0; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            l[batch_count * i + j]  = dl[batch_stride * i + j];
            u0[batch_count * i + j] = d[batch_stride * i + j];
            u1[batch_count * i + j] = du[batch_stride * i + j];
        }
    }

    // LU decomposition
    for(rocsparse_int i = 0; i < m - 1; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            T ak_1 = l[batch_count * (i + 1) + j];
            T bk   = u0[batch_count * i + j];

            if(std::abs(bk) < std::abs(ak_1))
            {
                T bk_1 = u0[batch_count * (i + 1) + j];
                T ck   = u1[batch_count * i + j];
                T ck_1 = u1[batch_count * (i + 1) + j];
                T dk   = u2[batch_count * i + j];

                u0[batch_count * i + j] = ak_1;
                u1[batch_count * i + j] = bk_1;
                u2[batch_count * i + j] = ck_1;

                u0[batch_count * (i + 1) + j] = ck;
                u1[batch_count * (i + 1) + j] = dk;

                rocsparse_int pk             = p[batch_count * i + j];
                p[batch_count * i + j]       = i + 1;
                p[batch_count * (i + 1) + j] = pk;

                T xk                          = x[batch_stride * i + j];
                x[batch_stride * i + j]       = x[batch_stride * (i + 1) + j];
                x[batch_stride * (i + 1) + j] = xk;

                T lk_1                       = bk / ak_1;
                l[batch_count * (i + 1) + j] = lk_1;

                u0[batch_count * (i + 1) + j]
                    = u0[batch_count * (i + 1) + j] - lk_1 * u1[batch_count * i + j];
                u1[batch_count * (i + 1) + j]
                    = u1[batch_count * (i + 1) + j] - lk_1 * u2[batch_count * i + j];
            }
            else
            {
                p[batch_count * (i + 1) + j] = i + 1;

                T lk_1                       = ak_1 / bk;
                l[batch_count * (i + 1) + j] = lk_1;

                u0[batch_count * (i + 1) + j]
                    = u0[batch_count * (i + 1) + j] - lk_1 * u1[batch_count * i + j];
                u1[batch_count * (i + 1) + j]
                    = u1[batch_count * (i + 1) + j] - lk_1 * u2[batch_count * i + j];
            }
        }
    }

    // Forward elimination (L * x_new = x_old)
    std::vector<rocsparse_int> start(batch_count, 0);
    for(rocsparse_int i = 1; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            if(p[batch_count * i + j] <= i) // no pivoting occured, sum up result
            {
                T temp = static_cast<T>(0);
                for(rocsparse_int s = start[j]; s < i; s++)
                {
                    temp = temp - l[batch_count * (s + 1) + j] * x[batch_stride * s + j];
                }
                x[batch_stride * i + j] = x[batch_stride * i + j] + temp;
                start[j] += i - start[j];
            }
        }
    }

    // backward substitution (U * x_newest = x_new)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / u0[batch_count * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - u1[batch_count * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / u0[batch_count * (m - 2) + j];
    }

    for(rocsparse_int i = m - 3; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j]
                = (x[batch_stride * i + j] - u1[batch_count * i + j] * x[batch_stride * (i + 1) + j]
                   - u2[batch_count * i + j] * x[batch_stride * (i + 2) + j])
                  / u0[batch_count * i + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch_qr(rocsparse_int m,
                                    const T*      dl,
                                    const T*      d,
                                    const T*      du,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    std::vector<T> r0(m * batch_count, 0);
    std::vector<T> r1(m * batch_count, 0);
    std::vector<T> r2(m * batch_count, 0);

    for(rocsparse_int i = 0; i < m; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            r0[batch_count * i + j] = d[batch_stride * i + j];
            r1[batch_count * i + j] = du[batch_stride * i + j];
        }
    }

    // Reduce A = Q*R where Q is orthonormal and R is upper triangular
    // This means when solving A * x          = b
    //                      => Q * R * x      = b
    //                      => Q' * Q * R * x = Q' * b
    //                      => R * x          = Q' * b
    // Because A is tri-diagonal, we use Givens rotations
    // Note on notation used here. I consider the A matrix to have form:
    // A = b0 c0 0  0  0
    //     a1 b1 c1 0  0
    //     0  a2 b2 c2 0
    //     0  0  a3 b3 c3
    //     0  0  0  a4 b4
    for(rocsparse_int i = 0; i < m - 1; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            T ak_1 = dl[batch_stride * (i + 1) + j];
            T bk   = r0[batch_count * i + j];
            T bk_1 = r0[batch_count * (i + 1) + j];
            T ck   = r1[batch_count * i + j];
            T ck_1 = r1[batch_count * (i + 1) + j];

            T radius = std::sqrt(std::abs(bk * rocsparse_conj(bk) + ak_1 * rocsparse_conj(ak_1)));

            // Apply Givens rotation
            // | cos  sin | |bk    ck   0   |
            // |-sin  cos | |ak_1  bk_1 ck_1|
            T cos_theta = rocsparse_conj(bk) / radius;
            T sin_theta = rocsparse_conj(ak_1) / radius;

            r0[batch_count * i + j] = std::fma(bk, cos_theta, ak_1 * sin_theta);
            r0[batch_count * (i + 1) + j]
                = std::fma(-ck, rocsparse_conj(sin_theta), bk_1 * rocsparse_conj(cos_theta));
            r1[batch_count * i + j]       = std::fma(ck, cos_theta, bk_1 * sin_theta);
            r1[batch_count * (i + 1) + j] = ck_1 * rocsparse_conj(cos_theta);
            r2[batch_count * i + j]       = ck_1 * sin_theta;

            // Apply Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk                    = x[batch_stride * i + j];
            T xk_1                  = x[batch_stride * (i + 1) + j];
            x[batch_stride * i + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
            x[batch_stride * (i + 1) + j]
                = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
        }
    }

    // Backward substitution on upper triangular R * x = x
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / r0[batch_count * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - r1[batch_count * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / r0[batch_count * (m - 2) + j];
    }

    for(rocsparse_int i = m - 3; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j]
                = (x[batch_stride * i + j] - r1[batch_count * i + j] * x[batch_stride * (i + 1) + j]
                   - r2[batch_count * i + j] * x[batch_stride * (i + 2) + j])
                  / r0[batch_count * i + j];
        }
    }
}

template <typename T>
void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 const T*                       dl,
                                 const T*                       d,
                                 const T*                       du,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(algo)
    {
    case rocsparse_gtsv_interleaved_alg_thomas:
    {
        host_gtsv_interleaved_batch_thomas(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    case rocsparse_gtsv_interleaved_alg_lu:
    {
        host_gtsv_interleaved_batch_lu(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    case rocsparse_gtsv_interleaved_alg_default:
    case rocsparse_gtsv_interleaved_alg_qr:
    {
        host_gtsv_interleaved_batch_qr(m, dl, d, du, x, batch_count, batch_stride);
        break;
    }
    }
}

template <typename T>
void host_gpsv_interleaved_batch_qr(rocsparse_int m,
                                    T*            ds,
                                    T*            dl,
                                    T*            d,
                                    T*            du,
                                    T*            dw,
                                    T*            x,
                                    rocsparse_int batch_count,
                                    rocsparse_int batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    std::vector<T> r3(m * batch_count, 0);
    std::vector<T> r4(m * batch_count, 0);

    // Reduce A = Q*R where Q is orthonormal and R is upper triangular
    // This means when solving A * x          = b
    //                      => Q * R * x      = b
    //                      => Q' * Q * R * x = Q' * b
    //                      => R * x          = Q' * b
    // Because A is penta-diagonal, we use Givens rotations
    // Note on notation used here. I consider the A matrix to have form:
    // A = d0 u0 w0  0  0
    //     l1 d1 u1 w1  0
    //     s2  l2 d2 u2 w2
    //     0  s3  l3 d3 u3
    //     0  0  s4  l4 d4
    for(rocsparse_int i = 0; i < m - 2; i++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            // For penta diagonal matrices, need to apply two givens rotations to remove lower and lower - 1 entries
            T radius    = static_cast<T>(0);
            T cos_theta = static_cast<T>(0);
            T sin_theta = static_cast<T>(0);

            // Apply first Givens rotation
            // | cos  sin | |lk_1 dk_1 uk_1 wk_1 0   |
            // |-sin  cos | |sk_2 lk_2 dk_2 uk_2 wk_2|
            T sk_2 = ds[batch_stride * (i + 2) + j];
            T lk_1 = dl[batch_stride * (i + 1) + j];
            T lk_2 = dl[batch_stride * (i + 2) + j];
            T dk_1 = d[batch_stride * (i + 1) + j];
            T dk_2 = d[batch_stride * (i + 2) + j];
            T uk_1 = du[batch_stride * (i + 1) + j];
            T uk_2 = du[batch_stride * (i + 2) + j];
            T wk_1 = dw[batch_stride * (i + 1) + j];
            T wk_2 = dw[batch_stride * (i + 2) + j];

            radius = std::sqrt(
                std::abs(std::fma(lk_1, rocsparse_conj(lk_1), sk_2 * rocsparse_conj(sk_2))));
            cos_theta = rocsparse_conj(lk_1) / radius;
            sin_theta = rocsparse_conj(sk_2) / radius;

            T dlk_1_new = std::fma(lk_1, cos_theta, sk_2 * sin_theta);
            T dk_1_new  = std::fma(dk_1, cos_theta, lk_2 * sin_theta);
            T duk_1_new = std::fma(uk_1, cos_theta, dk_2 * sin_theta);
            T dwk_1_new = std::fma(wk_1, cos_theta, uk_2 * sin_theta);

            dl[batch_stride * (i + 1) + j] = dlk_1_new;
            dl[batch_stride * (i + 2) + j]
                = std::fma(-dk_1, rocsparse_conj(sin_theta), lk_2 * rocsparse_conj(cos_theta));
            d[batch_stride * (i + 1) + j] = dk_1_new;
            d[batch_stride * (i + 2) + j]
                = std::fma(-uk_1, rocsparse_conj(sin_theta), dk_2 * rocsparse_conj(cos_theta));
            du[batch_stride * (i + 1) + j] = duk_1_new;
            du[batch_stride * (i + 2) + j]
                = std::fma(-wk_1, rocsparse_conj(sin_theta), uk_2 * rocsparse_conj(cos_theta));
            dw[batch_stride * (i + 1) + j] = dwk_1_new;
            dw[batch_stride * (i + 2) + j] = wk_2 * rocsparse_conj(cos_theta);
            r3[batch_count * (i + 1) + j]  = wk_2 * sin_theta;

            // Apply first Givens rotation to rhs vector
            // | cos  sin | |xk_1|
            // |-sin  cos | |xk_2|
            T xk_1                        = x[batch_stride * (i + 1) + j];
            T xk_2                        = x[batch_stride * (i + 2) + j];
            x[batch_stride * (i + 1) + j] = std::fma(xk_1, cos_theta, xk_2 * sin_theta);
            x[batch_stride * (i + 2) + j]
                = std::fma(-xk_1, rocsparse_conj(sin_theta), xk_2 * rocsparse_conj(cos_theta));

            // Apply second Givens rotation
            // | cos  sin | |dk   uk   wk   rk   0   |
            // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
            lk_1   = dlk_1_new;
            T dk   = d[batch_stride * i + j];
            dk_1   = dk_1_new;
            T uk   = du[batch_stride * i + j];
            uk_1   = duk_1_new;
            T wk   = dw[batch_stride * i + j];
            wk_1   = dwk_1_new;
            T rk   = r3[batch_count * i + j];
            T rk_1 = r3[batch_count * (i + 1) + j];

            radius = std::sqrt(
                std::abs(std::fma(dk, rocsparse_conj(dk), lk_1 * rocsparse_conj(lk_1))));
            cos_theta = rocsparse_conj(dk) / radius;
            sin_theta = rocsparse_conj(lk_1) / radius;

            d[batch_stride * i + j] = std::fma(dk, cos_theta, lk_1 * sin_theta);
            d[batch_stride * (i + 1) + j]
                = std::fma(-uk, rocsparse_conj(sin_theta), dk_1 * rocsparse_conj(cos_theta));
            du[batch_stride * i + j] = std::fma(uk, cos_theta, dk_1 * sin_theta);
            du[batch_stride * (i + 1) + j]
                = std::fma(-wk, rocsparse_conj(sin_theta), uk_1 * rocsparse_conj(cos_theta));
            dw[batch_stride * i + j] = std::fma(wk, cos_theta, uk_1 * sin_theta);
            dw[batch_stride * (i + 1) + j]
                = std::fma(-rk, rocsparse_conj(sin_theta), wk_1 * rocsparse_conj(cos_theta));
            r3[batch_count * i + j]       = std::fma(rk, cos_theta, wk_1 * sin_theta);
            r3[batch_count * (i + 1) + j] = rk_1 * rocsparse_conj(cos_theta);
            r4[batch_count * i + j]       = rk_1 * sin_theta;

            // Apply second Givens rotation to rhs vector
            // | cos  sin | |xk  |
            // |-sin  cos | |xk_1|
            T xk                    = x[batch_stride * i + j];
            xk_1                    = x[batch_stride * (i + 1) + j];
            x[batch_stride * i + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
            x[batch_stride * (i + 1) + j]
                = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
        }
    }

    // Apply last givens rotation
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        // Apply last Givens rotation
        // | cos  sin | |dk   uk   wk   rk   0   |
        // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
        T lk_1 = dl[batch_stride * (m - 1) + j];
        T dk   = d[batch_stride * (m - 2) + j];
        T dk_1 = d[batch_stride * (m - 1) + j];
        T uk   = du[batch_stride * (m - 2) + j];
        T uk_1 = du[batch_stride * (m - 1) + j];
        T wk   = dw[batch_stride * (m - 2) + j];
        T wk_1 = dw[batch_stride * (m - 1) + j];
        T rk   = r3[batch_count * (m - 2) + j];
        T rk_1 = r3[batch_count * (m - 1) + j];

        T radius
            = std::sqrt(std::abs(std::fma(dk, rocsparse_conj(dk), lk_1 * rocsparse_conj(lk_1))));
        T cos_theta = rocsparse_conj(dk) / radius;
        T sin_theta = rocsparse_conj(lk_1) / radius;

        d[batch_stride * (m - 2) + j] = std::fma(dk, cos_theta, lk_1 * sin_theta);
        d[batch_stride * (m - 1) + j]
            = std::fma(-uk, rocsparse_conj(sin_theta), dk_1 * rocsparse_conj(cos_theta));
        du[batch_stride * (m - 2) + j] = std::fma(uk, cos_theta, dk_1 * sin_theta);
        du[batch_stride * (m - 1) + j]
            = std::fma(-wk, rocsparse_conj(sin_theta), uk_1 * rocsparse_conj(cos_theta));
        dw[batch_stride * (m - 2) + j] = std::fma(wk, cos_theta, uk_1 * sin_theta);
        dw[batch_stride * (m - 1) + j]
            = std::fma(-rk, rocsparse_conj(sin_theta), wk_1 * rocsparse_conj(cos_theta));
        r3[batch_count * (m - 2) + j] = std::fma(rk, cos_theta, wk_1 * sin_theta);
        r3[batch_count * (m - 1) + j] = rk_1 * rocsparse_conj(cos_theta);
        r4[batch_count * (m - 2) + j] = rk_1 * sin_theta;

        // Apply last Givens rotation to rhs vector
        // | cos  sin | |xk  |
        // |-sin  cos | |xk_1|
        T xk                          = x[batch_stride * (m - 2) + j];
        T xk_1                        = x[batch_stride * (m - 1) + j];
        x[batch_stride * (m - 2) + j] = std::fma(xk, cos_theta, xk_1 * sin_theta);
        x[batch_stride * (m - 1) + j]
            = std::fma(-xk, rocsparse_conj(sin_theta), xk_1 * rocsparse_conj(cos_theta));
    }

    // Backward substitution on upper triangular R * x = x
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int j = 0; j < batch_count; j++)
    {
        x[batch_stride * (m - 1) + j]
            = x[batch_stride * (m - 1) + j] / d[batch_stride * (m - 1) + j];
        x[batch_stride * (m - 2) + j]
            = (x[batch_stride * (m - 2) + j]
               - du[batch_stride * (m - 2) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 2) + j];

        x[batch_stride * (m - 3) + j]
            = (x[batch_stride * (m - 3) + j]
               - du[batch_stride * (m - 3) + j] * x[batch_stride * (m - 2) + j]
               - dw[batch_stride * (m - 3) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 3) + j];

        x[batch_stride * (m - 4) + j]
            = (x[batch_stride * (m - 4) + j]
               - du[batch_stride * (m - 4) + j] * x[batch_stride * (m - 3) + j]
               - dw[batch_stride * (m - 4) + j] * x[batch_stride * (m - 2) + j]
               - r3[batch_count * (m - 4) + j] * x[batch_stride * (m - 1) + j])
              / d[batch_stride * (m - 4) + j];
    }

    for(rocsparse_int i = m - 5; i >= 0; i--)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            x[batch_stride * i + j] = (x[batch_stride * i + j]
                                       - du[batch_stride * i + j] * x[batch_stride * (i + 1) + j]
                                       - dw[batch_stride * i + j] * x[batch_stride * (i + 2) + j]
                                       - r3[batch_count * i + j] * x[batch_stride * (i + 3) + j]
                                       - r4[batch_count * i + j] * x[batch_stride * (i + 4) + j])
                                      / d[batch_stride * i + j];
        }
    }
}

template <typename T>
void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 T*                             ds,
                                 T*                             dl,
                                 T*                             d,
                                 T*                             du,
                                 T*                             dw,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    switch(algo)
    {
    case rocsparse_gpsv_interleaved_alg_default:
    case rocsparse_gpsv_interleaved_alg_qr:
    {
        host_gpsv_interleaved_batch_qr(m, ds, dl, d, du, dw, x, batch_count, batch_stride);
        break;
    }
    }
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
rocsparse_status host_nnz(rocsparse_direction dirA,
                          rocsparse_int       m,
                          rocsparse_int       n,
                          const T*            A,
                          int64_t             lda,
                          rocsparse_int*      nnz_per_row_columns,
                          rocsparse_int*      nnz_total_dev_host_ptr)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int mn = (dirA == rocsparse_direction_row) ? m : n;
    for(rocsparse_int j = 0; j < mn; ++j)
    {
        nnz_per_row_columns[j] = 0;
    }

    for(rocsparse_int j = 0; j < n; ++j)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            if(A[j * lda + i] != 0)
            {
                if(dirA == rocsparse_direction_row)
                {
                    nnz_per_row_columns[i] += 1;
                }
                else
                {
                    nnz_per_row_columns[j] += 1;
                }
            }
        }
    }

    nnz_total_dev_host_ptr[0] = 0;
    for(rocsparse_int j = 0; j < mn; ++j)
    {
        nnz_total_dev_host_ptr[0] += nnz_per_row_columns[j];
    }

    return rocsparse_status_success;
}

template <typename T>
void host_prune_dense2csr(rocsparse_int               m,
                          rocsparse_int               n,
                          const std::vector<T>&       A,
                          int64_t                     lda,
                          rocsparse_index_base        base,
                          T                           threshold,
                          rocsparse_int&              nnz,
                          std::vector<T>&             csr_val,
                          std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    csr_row_ptr.resize(m + 1, 0);
    csr_row_ptr[0] = base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < m; i++)
    {
        for(rocsparse_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_row_ptr[i + 1]++;
            }
        }
    }

    for(rocsparse_int i = 1; i <= m; i++)
    {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    nnz = csr_row_ptr[m] - csr_row_ptr[0];

    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    rocsparse_int index = 0;
    for(rocsparse_int i = 0; i < m; i++)
    {
        for(rocsparse_int j = 0; j < n; j++)
        {
            if(std::abs(A[lda * j + i]) > threshold)
            {
                csr_val[index]     = A[lda * j + i];
                csr_col_ind[index] = j + base;

                index++;
            }
        }
    }
}

template <typename T>
void host_prune_dense2csr_by_percentage(rocsparse_int               m,
                                        rocsparse_int               n,
                                        const std::vector<T>&       A,
                                        int64_t                     lda,
                                        rocsparse_index_base        base,
                                        T                           percentage,
                                        rocsparse_int&              nnz,
                                        std::vector<T>&             csr_val,
                                        std::vector<rocsparse_int>& csr_row_ptr,
                                        std::vector<rocsparse_int>& csr_col_ind)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int nnz_A = m * n;
    rocsparse_int pos   = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos                 = std::min(pos, nnz_A - 1);
    pos                 = std::max(pos, static_cast<rocsparse_int>(0));

    std::vector<T> sorted_A(m * n);
    for(rocsparse_int i = 0; i < n; i++)
    {
        for(rocsparse_int j = 0; j < m; j++)
        {
            sorted_A[m * i + j] = std::abs(A[lda * i + j]);
        }
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = sorted_A[pos];
    host_prune_dense2csr<T>(m, n, A, lda, base, threshold, nnz, csr_val, csr_row_ptr, csr_col_ind);
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_dense2csx(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    const T*             A,
                    int64_t              ld,
                    rocsparse_order      order,
                    const I*             nnz_per_row_columns,
                    T*                   csx_val,
                    I*                   csx_row_col_ptr,
                    J*                   csx_col_row_ind)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    static constexpr T s_zero = {};
    J                  len    = (rocsparse_direction_row == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(J i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case rocsparse_direction_column:
    {
        for(J j = 0; j < n; ++j)
        {
            for(J i = 0; i < m; ++i)
            {
                if(order == rocsparse_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = i + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = i + base;
                    }
                }
            }
        }

        break;
    }

    case rocsparse_direction_row:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csx_row_ptr_A to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csx_row_ptr_A to its initial state)
        //
        for(J i = 0; i < m; ++i)
        {
            for(J j = 0; j < n; ++j)
            {
                if(order == rocsparse_order_column)
                {
                    if(A[j * ld + i] != s_zero)
                    {
                        *csx_val++         = A[j * ld + i];
                        *csx_col_row_ind++ = j + base;
                    }
                }
                else
                {
                    if(A[i * ld + j] != s_zero)
                    {
                        *csx_val++         = A[i * ld + j];
                        *csx_col_row_ind++ = j + base;
                    }
                }
            }
        }

        break;
    }
    }
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_csx2dense(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    rocsparse_order      order,
                    const T*             csx_val,
                    const I*             csx_row_col_ptr,
                    const J*             csx_col_row_ind,
                    T*                   A,
                    int64_t              ld)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(order == rocsparse_order_column)
    {
        for(J col = 0; col < n; ++col)
        {
            for(J row = 0; row < m; ++row)
            {
                A[row + ld * col] = static_cast<T>(0);
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            for(J col = 0; col < n; ++col)
            {
                A[col + ld * row] = static_cast<T>(0);
            }
        }
    }

    if(DIRA == rocsparse_direction_column)
    {
        for(J col = 0; col < n; ++col)
        {
            I start = csx_row_col_ptr[col] - base;
            I end   = csx_row_col_ptr[col + 1] - base;

            if(order == rocsparse_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {
                    A[col + ld * (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
    else
    {
        for(J row = 0; row < m; ++row)
        {
            I start = csx_row_col_ptr[row] - base;
            I end   = csx_row_col_ptr[row + 1] - base;

            if(order == rocsparse_order_column)
            {
                for(I at = start; at < end; ++at)
                {
                    A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
                }
            }
            else
            {
                for(I at = start; at < end; ++at)
                {

                    A[row * ld + (csx_col_row_ind[at] - base)] = csx_val[at];
                }
            }
        }
    }
}

template <typename I, typename T>
void host_dense_to_coo(I                     m,
                       I                     n,
                       rocsparse_index_base  base,
                       const std::vector<T>& A,
                       int64_t               ld,
                       rocsparse_order       order,
                       const std::vector<I>& nnz_per_row,
                       std::vector<T>&       coo_val,
                       std::vector<I>&       coo_row_ind,
                       std::vector<I>&       coo_col_ind)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Find number of non-zeros in dense matrix
    int64_t nnz = 0;
    for(I i = 0; i < m; ++i)
    {
        nnz += nnz_per_row[i];
    }

    coo_val.resize(nnz, static_cast<T>(0));
    coo_row_ind.resize(nnz, 0);
    coo_col_ind.resize(nnz, 0);

    // Fill COO matrix
    int64_t index = 0;
    for(I i = 0; i < m; i++)
    {
        for(I j = 0; j < n; j++)
        {
            if(order == rocsparse_order_column)
            {
                if(A[ld * j + i] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * j + i];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
            else
            {
                if(A[ld * i + j] != static_cast<T>(0))
                {
                    coo_val[index]     = A[ld * i + j];
                    coo_row_ind[index] = i + base;
                    coo_col_ind[index] = j + base;

                    index++;
                }
            }
        }
    }
}

template <typename I, typename T>
void host_coo_to_dense(I                     m,
                       I                     n,
                       int64_t               nnz,
                       rocsparse_index_base  base,
                       const std::vector<T>& coo_val,
                       const std::vector<I>& coo_row_ind,
                       const std::vector<I>& coo_col_ind,
                       std::vector<T>&       A,
                       int64_t               ld,
                       rocsparse_order       order)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    I nm = order == rocsparse_order_column ? n : m;

    A.resize(ld * nm);

    if(order == rocsparse_order_column)
    {
        for(I i = 0; i < n; i++)
        {
            for(I j = 0; j < m; j++)
            {
                A[ld * i + j] = static_cast<T>(0);
            }
        }
    }
    else
    {
        for(I j = 0; j < m; j++)
        {
            for(I i = 0; i < n; i++)
            {
                A[ld * j + i] = static_cast<T>(0);
            }
        }
    }

    for(int64_t i = 0; i < nnz; i++)
    {
        I row = coo_row_ind[i] - base;
        I col = coo_col_ind[i] - base;
        T val = coo_val[i];

        if(order == rocsparse_order_column)
        {
            A[ld * col + row] = val;
        }
        else
        {
            A[ld * row + col] = val;
        }
    }
}

template <typename I, typename J, typename T>
void host_csr_to_csc(J                    M,
                     J                    N,
                     I                    nnz,
                     const I*             csr_row_ptr,
                     const J*             csr_col_ind,
                     const T*             csr_val,
                     std::vector<J>&      csc_row_ind,
                     std::vector<I>&      csc_col_ptr,
                     std::vector<T>&      csc_val,
                     rocsparse_action     action,
                     rocsparse_index_base base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(J i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            J col = csr_col_ind[j] - base;
            I idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(J i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    if(csc_col_ptr.size() > 0)
    {
        csc_col_ptr[0] = base;
    }
}

template <typename T>
void host_bsr_to_csr(rocsparse_direction               direction,
                     rocsparse_int                     mb,
                     rocsparse_int                     nb,
                     rocsparse_int                     nnzb,
                     const std::vector<T>&             bsr_val,
                     const std::vector<rocsparse_int>& bsr_row_ptr,
                     const std::vector<rocsparse_int>& bsr_col_ind,
                     rocsparse_int                     block_dim,
                     rocsparse_index_base              bsr_base,
                     std::vector<T>&                   csr_val,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     std::vector<rocsparse_int>&       csr_col_ind,
                     rocsparse_index_base              csr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return host_gebsr_to_csr(direction,
                             mb,
                             nb,
                             nnzb,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             block_dim,
                             bsr_base,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             csr_base);
}

template <typename T>
void host_csr_to_bsr(rocsparse_direction               direction,
                     rocsparse_int                     m,
                     rocsparse_int                     n,
                     rocsparse_int                     nnz,
                     const std::vector<T>&             csr_val,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     rocsparse_int                     block_dim,
                     rocsparse_index_base              csr_base,
                     std::vector<T>&                   bsr_val,
                     std::vector<rocsparse_int>&       bsr_row_ptr,
                     std::vector<rocsparse_int>&       bsr_col_ind,
                     rocsparse_index_base              bsr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return host_csr_to_gebsr(direction,
                             m,
                             n,
                             nnz,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             block_dim,
                             block_dim,
                             csr_base,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             bsr_base);
}

template <typename T>
void host_csr_to_gebsr(rocsparse_direction               direction,
                       rocsparse_int                     m,
                       rocsparse_int                     n,
                       rocsparse_int                     nnz,
                       const std::vector<T>&             csr_val,
                       const std::vector<rocsparse_int>& csr_row_ptr,
                       const std::vector<rocsparse_int>& csr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              csr_base,
                       std::vector<T>&                   bsr_val,
                       std::vector<rocsparse_int>&       bsr_row_ptr,
                       std::vector<rocsparse_int>&       bsr_col_ind,
                       rocsparse_index_base              bsr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;

    bsr_row_ptr.resize(mb + 1, 0);

    std::vector<rocsparse_int> temp(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz; i++)
    {
        temp[i] = (csr_col_ind[i] - csr_base) / col_block_dim;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int frow = row_block_dim * i;
        rocsparse_int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        rocsparse_int start = csr_row_ptr[frow] - csr_base;
        rocsparse_int end   = csr_row_ptr[lrow] - csr_base;

        std::sort(temp.begin() + start, temp.begin() + end);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < mb; i++)
    {
        rocsparse_int frow = row_block_dim * i;
        rocsparse_int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        rocsparse_int start = csr_row_ptr[frow] - csr_base;
        rocsparse_int end   = csr_row_ptr[lrow] - csr_base;

        rocsparse_int col   = -1;
        rocsparse_int count = 0;
        for(rocsparse_int j = start; j < end; j++)
        {
            if(temp[j] > col)
            {
                col                 = temp[j];
                temp[j]             = -1;
                temp[start + count] = col;
                count++;
            }
            else
            {
                temp[j] = -1;
            }
        }

        bsr_row_ptr[i + 1] = count;
    }

    // fill GEBSR row pointer array
    bsr_row_ptr[0] = bsr_base;
    for(rocsparse_int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];
    bsr_col_ind.resize(nnzb);
    bsr_val.resize(nnzb * row_block_dim * col_block_dim, 0);

    // fill GEBSR col indices array
    {
        rocsparse_int index = 0;
        for(rocsparse_int i = 0; i < nnz; i++)
        {
            if(temp[i] != -1)
            {
                bsr_col_ind[index] = temp[i] + bsr_base;
                index++;
            }
        }
    }

    // fill GEBSR values array
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < m; i++)
    {
        rocsparse_int start = csr_row_ptr[i] - csr_base;
        rocsparse_int end   = csr_row_ptr[i + 1] - csr_base;

        rocsparse_int bstart = bsr_row_ptr[i / row_block_dim] - bsr_base;
        rocsparse_int bend   = bsr_row_ptr[i / row_block_dim + 1] - bsr_base;

        rocsparse_int local_row = i % row_block_dim;

        for(rocsparse_int j = start; j < end; j++)
        {
            rocsparse_int col = csr_col_ind[j] - csr_base;

            rocsparse_int local_col = col % col_block_dim;

            rocsparse_int index = 0;
            for(rocsparse_int k = bstart; k < bend; k++)
            {
                if(bsr_col_ind[k] - bsr_base == col / col_block_dim)
                {
                    index  = k;
                    bstart = k;
                    break;
                }
            }

            if(direction == rocsparse_direction_row)
            {
                bsr_val[row_block_dim * col_block_dim * index + col_block_dim * local_row
                        + local_col]
                    = csr_val[j];
            }
            else
            {
                bsr_val[row_block_dim * col_block_dim * index + row_block_dim * local_col
                        + local_row]
                    = csr_val[j];
            }
        }
    }
}

template <typename T>
void host_gebsr_to_gebsc(rocsparse_int                     Mb,
                         rocsparse_int                     Nb,
                         rocsparse_int                     nnzb,
                         const std::vector<rocsparse_int>& bsr_row_ptr,
                         const std::vector<rocsparse_int>& bsr_col_ind,
                         const std::vector<T>&             bsr_val,
                         rocsparse_int                     row_block_dim,
                         rocsparse_int                     col_block_dim,
                         std::vector<rocsparse_int>&       bsc_row_ind,
                         std::vector<rocsparse_int>&       bsc_col_ptr,
                         std::vector<T>&                   bsc_val,
                         rocsparse_action                  action,
                         rocsparse_index_base              base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(Nb + 1, 0);
    bsc_val.resize(nnzb * row_block_dim * col_block_dim);

    const rocsparse_int block_shift = row_block_dim * col_block_dim;

    //
    // Determine nnz per column
    //
    for(rocsparse_int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - base];
    }

    // Scan
    for(rocsparse_int i = 0; i < Nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < Mb; ++i)
    {
        const rocsparse_int row_begin = bsr_row_ptr[i] - base;
        const rocsparse_int row_end   = bsr_row_ptr[i + 1] - base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            const rocsparse_int col = bsr_col_ind[j] - base;
            const rocsparse_int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + base;
            for(rocsparse_int k = 0; k < block_shift; ++k)
            {
                bsc_val[idx * block_shift + k] = bsr_val[j * block_shift + k];
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = Nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + base;
    }

    bsc_col_ptr[0] = base;
}

template <typename T>
void host_gebsr_to_csr(rocsparse_direction               direction,
                       rocsparse_int                     mb,
                       rocsparse_int                     nb,
                       rocsparse_int                     nnzb,
                       const std::vector<T>&             bsr_val,
                       const std::vector<rocsparse_int>& bsr_row_ptr,
                       const std::vector<rocsparse_int>& bsr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              bsr_base,
                       std::vector<T>&                   csr_val,
                       std::vector<rocsparse_int>&       csr_row_ptr,
                       std::vector<rocsparse_int>&       csr_col_ind,
                       rocsparse_index_base              csr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int m   = mb * row_block_dim;
    size_t        nnz = size_t(nnzb) * row_block_dim * col_block_dim;

    csr_row_ptr.resize(m + 1);
    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    csr_row_ptr[0] = csr_base;

    if(nb == 0)
    {
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                rocsparse_int row    = i * row_block_dim + r;
                csr_row_ptr[row + 1] = csr_base;
            }
        }

        return;
    }

    for(rocsparse_int i = 0; i < mb; ++i)
    {
        rocsparse_int start = bsr_row_ptr[i] - bsr_base;
        rocsparse_int end   = bsr_row_ptr[i + 1] - bsr_base;

        for(rocsparse_int k = start; k < end; ++k)
        {
            rocsparse_int j = bsr_col_ind[k] - bsr_base;

            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                for(rocsparse_int c = 0; c < col_block_dim; ++c)
                {
                    rocsparse_int col = col_block_dim * j + c;

                    rocsparse_int index = start * row_block_dim * col_block_dim
                                          + (end - start) * col_block_dim * r
                                          + (k - start) * col_block_dim + c;

                    csr_col_ind[index] = col + csr_base;
                    if(direction == rocsparse_direction_row)
                    {
                        csr_val[index]
                            = bsr_val[k * row_block_dim * col_block_dim + col_block_dim * r + c];
                    }
                    else
                    {
                        csr_val[index]
                            = bsr_val[k * row_block_dim * col_block_dim + row_block_dim * c + r];
                    }
                }
            }
        }

        for(rocsparse_int r = 0; r < row_block_dim; ++r)
        {
            rocsparse_int row    = i * row_block_dim + r;
            csr_row_ptr[row + 1] = csr_row_ptr[row] + (end - start) * col_block_dim;
        }
    }
}

template <typename T>
void host_gebsr_to_gebsr(rocsparse_direction               direction,
                         rocsparse_int                     mb,
                         rocsparse_int                     nb,
                         rocsparse_int                     nnzb,
                         const std::vector<T>&             bsr_val_A,
                         const std::vector<rocsparse_int>& bsr_row_ptr_A,
                         const std::vector<rocsparse_int>& bsr_col_ind_A,
                         rocsparse_int                     row_block_dim_A,
                         rocsparse_int                     col_block_dim_A,
                         rocsparse_index_base              base_A,
                         std::vector<T>&                   bsr_val_C,
                         std::vector<rocsparse_int>&       bsr_row_ptr_C,
                         std::vector<rocsparse_int>&       bsr_col_ind_C,
                         rocsparse_int                     row_block_dim_C,
                         rocsparse_int                     col_block_dim_C,
                         rocsparse_index_base              base_C)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int m = mb * row_block_dim_A;
    rocsparse_int n = nb * col_block_dim_A;

    // convert GEBSR to CSR format
    std::vector<rocsparse_int> csr_row_ptr;
    std::vector<rocsparse_int> csr_col_ind;
    std::vector<T>             csr_val;

    host_gebsr_to_csr(direction,
                      mb,
                      nb,
                      nnzb,
                      bsr_val_A,
                      bsr_row_ptr_A,
                      bsr_col_ind_A,
                      row_block_dim_A,
                      col_block_dim_A,
                      base_A,
                      csr_val,
                      csr_row_ptr,
                      csr_col_ind,
                      rocsparse_index_base_zero);

    rocsparse_int nnz = csr_row_ptr[m] - csr_row_ptr[0];

    // convert CSR to GEBSR format
    host_csr_to_gebsr(direction,
                      m,
                      n,
                      nnz,
                      csr_val,
                      csr_row_ptr,
                      csr_col_ind,
                      row_block_dim_C,
                      col_block_dim_C,
                      rocsparse_index_base_zero,
                      bsr_val_C,
                      bsr_row_ptr_C,
                      bsr_col_ind_C,
                      base_C);
}

template <typename T>
void host_bsr_to_bsc(rocsparse_int               mb,
                     rocsparse_int               nb,
                     rocsparse_int               nnzb,
                     rocsparse_int               bsr_dim,
                     const rocsparse_int*        bsr_row_ptr,
                     const rocsparse_int*        bsr_col_ind,
                     const T*                    bsr_val,
                     std::vector<rocsparse_int>& bsc_row_ind,
                     std::vector<rocsparse_int>& bsc_col_ptr,
                     std::vector<T>&             bsc_val,
                     rocsparse_index_base        bsr_base,
                     rocsparse_index_base        bsc_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(nb + 1, 0);
    bsc_val.resize(nnzb * bsr_dim * bsr_dim);

    // Determine nnz per column
    for(rocsparse_int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - bsr_base];
    }

    // Scan
    for(rocsparse_int i = 0; i < nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < mb; ++i)
    {
        rocsparse_int row_begin = bsr_row_ptr[i] - bsr_base;
        rocsparse_int row_end   = bsr_row_ptr[i + 1] - bsr_base;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col = bsr_col_ind[j] - bsr_base;
            rocsparse_int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + bsc_base;

            for(rocsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                for(rocsparse_int bj = 0; bj < bsr_dim; ++bj)
                {
                    bsc_val[bsr_dim * bsr_dim * idx + bi + bj * bsr_dim]
                        = bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj];
                }
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + bsc_base;
    }

    bsc_col_ptr[0] = bsc_base;
}

template <typename T>
void host_csr_to_hyb(rocsparse_int                     M,
                     rocsparse_int                     N,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     const std::vector<T>&             csr_val,
                     std::vector<rocsparse_int>&       ell_col_ind,
                     std::vector<T>&                   ell_val,
                     rocsparse_int&                    ell_width,
                     rocsparse_int&                    ell_nnz,
                     std::vector<rocsparse_int>&       coo_row_ind,
                     std::vector<rocsparse_int>&       coo_col_ind,
                     std::vector<T>&                   coo_val,
                     rocsparse_int&                    coo_nnz,
                     rocsparse_hyb_partition           part,
                     rocsparse_index_base              base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    ell_nnz = 0;
    coo_nnz = 0;

    if(M == 0 || N == 0)
    {
        ell_width = 0;
        return;
    }

    // Auto and user width
    if(part == rocsparse_hyb_partition_auto || part == rocsparse_hyb_partition_user)
    {
        // Determine ELL width
        ell_width = (part == rocsparse_hyb_partition_auto) ? (nnz - 1) / M + 1 : ell_width;

        // Determine COO nnz
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];

            if(row_nnz > ell_width)
            {
                coo_nnz += row_nnz - ell_width;
            }
        }
    }
    else if(part == rocsparse_hyb_partition_max)
    {
        ell_width = 0;

        // Determine max nnz per row
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
            ell_width             = std::max(ell_width, row_nnz);
        }
    }

    // ELL nnz
    ell_nnz = ell_width * M;

    // Allocate memory for HYB matrix
    if(ell_nnz > 0)
    {
        ell_col_ind.resize(ell_nnz);
        ell_val.resize(ell_nnz);
    }

    if(coo_nnz > 0)
    {
        coo_row_ind.resize(coo_nnz);
        coo_col_ind.resize(coo_nnz);
        coo_val.resize(coo_nnz);
    }

    // Fill HYB
    rocsparse_int coo_idx = 0;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int p         = 0;
        rocsparse_int row_begin = csr_row_ptr[i] - base;
        rocsparse_int row_end   = csr_row_ptr[i + 1] - base;
        rocsparse_int row_nnz   = row_end - row_begin;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            if(p < ell_width)
            {
                rocsparse_int idx = p++ * M + i;

                ell_col_ind[idx] = csr_col_ind[j];
                ell_val[idx]     = csr_val[j];
            }
            else
            {
                coo_row_ind[coo_idx] = i + base;
                coo_col_ind[coo_idx] = csr_col_ind[j];
                coo_val[coo_idx++]   = csr_val[j];
            }
        }

        for(rocsparse_int j = row_nnz; j < ell_width; ++j)
        {
            rocsparse_int idx = p++ * M + i;

            ell_col_ind[idx] = -1;
            ell_val[idx]     = static_cast<T>(0);
        }
    }
}

template <typename T>
void host_csr_to_csr_compress(rocsparse_int                     M,
                              rocsparse_int                     N,
                              rocsparse_int                     nnz,
                              const std::vector<rocsparse_int>& csr_row_ptr_A,
                              const std::vector<rocsparse_int>& csr_col_ind_A,
                              const std::vector<T>&             csr_val_A,
                              std::vector<rocsparse_int>&       csr_row_ptr_C,
                              std::vector<rocsparse_int>&       csr_col_ind_C,
                              std::vector<T>&                   csr_val_C,
                              rocsparse_index_base              base,
                              T                                 tol)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(M <= 0 || N <= 0)
    {
        return;
    }

    // find how many entries will be in each compressed CSR matrix row
    std::vector<rocsparse_int> nnz_per_row(M);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int start = csr_row_ptr_A[i] - base;
        rocsparse_int end   = csr_row_ptr_A[i + 1] - base;
        rocsparse_int count = 0;

        for(rocsparse_int j = start; j < end; j++)
        {
            if(std::abs(csr_val_A[j]) > std::real(tol)
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                count++;
            }
        }

        nnz_per_row[i] = count;
    }

    // add up total number of entries
    rocsparse_int nnz_C = 0;
    for(rocsparse_int i = 0; i < M; i++)
    {
        nnz_C += nnz_per_row[i];
    }

    //column indices and value arrays for compressed CSR matrix
    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    // fill in row pointer array for compressed CSR matrix
    csr_row_ptr_C.resize(M + 1);

    csr_row_ptr_C[0] = base;
    for(rocsparse_int i = 0; i < M; i++)
    {
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i] + nnz_per_row[i];
    }

    // fill in column indices and value arrays for compressed CSR matrix
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        rocsparse_int start = csr_row_ptr_A[i] - base;
        rocsparse_int end   = csr_row_ptr_A[i + 1] - base;
        rocsparse_int index = csr_row_ptr_C[i] - base;

        for(rocsparse_int j = start; j < end; j++)
        {
            if(std::abs(csr_val_A[j]) > std::real(tol)
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_col_ind_C[index] = csr_col_ind_A[j];
                csr_val_C[index]     = csr_val_A[j];
                index++;
            }
        }
    }
}

template <typename T>
void host_prune_csr_to_csr(rocsparse_int                     M,
                           rocsparse_int                     N,
                           rocsparse_int                     nnz_A,
                           const std::vector<rocsparse_int>& csr_row_ptr_A,
                           const std::vector<rocsparse_int>& csr_col_ind_A,
                           const std::vector<T>&             csr_val_A,
                           rocsparse_int&                    nnz_C,
                           std::vector<rocsparse_int>&       csr_row_ptr_C,
                           std::vector<rocsparse_int>&       csr_col_ind_C,
                           std::vector<T>&                   csr_val_C,
                           rocsparse_index_base              csr_base_A,
                           rocsparse_index_base              csr_base_C,
                           T                                 threshold)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    csr_row_ptr_C.resize(M + 1, 0);
    csr_row_ptr_C[0] = csr_base_C;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; i++)
    {
        for(rocsparse_int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A;
            j++)
        {
            if(std::abs(csr_val_A[j]) > threshold
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_row_ptr_C[i + 1]++;
            }
        }
    }

    for(rocsparse_int i = 1; i <= M; i++)
    {
        csr_row_ptr_C[i] += csr_row_ptr_C[i - 1];
    }

    nnz_C = csr_row_ptr_C[M] - csr_row_ptr_C[0];

    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    rocsparse_int index = 0;
    for(rocsparse_int i = 0; i < M; i++)
    {
        for(rocsparse_int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A;
            j++)
        {
            if(std::abs(csr_val_A[j]) > threshold
               && std::abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_col_ind_C[index] = (csr_col_ind_A[j] - csr_base_A) + csr_base_C;
                csr_val_C[index]     = csr_val_A[j];

                index++;
            }
        }
    }
}

template <typename T>
void host_prune_csr_to_csr_by_percentage(rocsparse_int                     M,
                                         rocsparse_int                     N,
                                         rocsparse_int                     nnz_A,
                                         const std::vector<rocsparse_int>& csr_row_ptr_A,
                                         const std::vector<rocsparse_int>& csr_col_ind_A,
                                         const std::vector<T>&             csr_val_A,
                                         rocsparse_int&                    nnz_C,
                                         std::vector<rocsparse_int>&       csr_row_ptr_C,
                                         std::vector<rocsparse_int>&       csr_col_ind_C,
                                         std::vector<T>&                   csr_val_C,
                                         rocsparse_index_base              csr_base_A,
                                         rocsparse_index_base              csr_base_C,
                                         T                                 percentage)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int pos = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos               = std::min(pos, nnz_A - 1);
    pos               = std::max(pos, static_cast<rocsparse_int>(0));

    std::vector<T> sorted_A(nnz_A);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < nnz_A; i++)
    {
        sorted_A[i] = std::abs(csr_val_A[i]);
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = nnz_A != 0 ? sorted_A[pos] : static_cast<T>(0);

    host_prune_csr_to_csr<T>(M,
                             N,
                             nnz_A,
                             csr_row_ptr_A,
                             csr_col_ind_A,
                             csr_val_A,
                             nnz_C,
                             csr_row_ptr_C,
                             csr_col_ind_C,
                             csr_val_C,
                             csr_base_A,
                             csr_base_C,
                             threshold);
}

template <typename T>
void host_ell_to_csr(rocsparse_int                     M,
                     rocsparse_int                     N,
                     const std::vector<rocsparse_int>& ell_col_ind,
                     const std::vector<T>&             ell_val,
                     rocsparse_int                     ell_width,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     std::vector<rocsparse_int>&       csr_col_ind,
                     std::vector<T>&                   csr_val,
                     rocsparse_int&                    csr_nnz,
                     rocsparse_index_base              ell_base,
                     rocsparse_index_base              csr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    csr_row_ptr.resize(M + 1, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            rocsparse_int idx = p * M + i;
            rocsparse_int col = ell_col_ind[idx] - ell_base;

            if(col >= 0 && col < N)
            {
                ++csr_row_ptr[i];
            }
        }
    }

    // Determine row pointers
    csr_nnz = csr_base;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int tmp = csr_row_ptr[i];
        csr_row_ptr[i]    = csr_nnz;
        csr_nnz += tmp;
    }

    csr_row_ptr[M] = csr_nnz;
    csr_nnz -= csr_base;

    // Allocate memory for columns and values
    csr_col_ind.resize(csr_nnz);
    csr_val.resize(csr_nnz);

    // Fill CSR structure
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(rocsparse_int i = 0; i < M; ++i)
    {
        rocsparse_int csr_idx = csr_row_ptr[i] - csr_base;

        for(rocsparse_int p = 0; p < ell_width; ++p)
        {
            rocsparse_int idx = p * M + i;
            rocsparse_int col = ell_col_ind[idx] - ell_base;

            if(col >= 0 && col < N)
            {
                csr_col_ind[csr_idx] = col + csr_base;
                csr_val[csr_idx]     = ell_val[idx];

                ++csr_idx;
            }
        }
    }
}

template <typename T>
void host_coosort_by_column(rocsparse_int               M,
                            rocsparse_int               nnz,
                            std::vector<rocsparse_int>& coo_row_ind,
                            std::vector<rocsparse_int>& coo_col_ind,
                            std::vector<T>&             coo_val)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Permutation vector
    std::vector<rocsparse_int> perm(nnz);

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::vector<rocsparse_int> tmp_row(nnz);
    std::vector<rocsparse_int> tmp_col(nnz);
    std::vector<T>             tmp_val(nnz);

    tmp_row = coo_row_ind;
    tmp_col = coo_col_ind;
    tmp_val = coo_val;

    // Sort
    std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
        if(tmp_col[a] < tmp_col[b])
        {
            return true;
        }
        else if(tmp_col[a] == tmp_col[b])
        {
            return (tmp_row[a] < tmp_row[b]);
        }
        else
        {
            return false;
        }
    });

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        coo_row_ind[i] = tmp_row[perm[i]];
        coo_col_ind[i] = tmp_col[perm[i]];
        coo_val[i]     = tmp_val[perm[i]];
    }
}

template <typename T>
void host_bsrpad_value(rocsparse_int m,
                       rocsparse_int mb,
                       rocsparse_int nnzb,
                       rocsparse_int block_dim,
                       T             value,
                       T* __restrict__ bsr_val,
                       const rocsparse_int* __restrict__ bsr_row_ptr,
                       const rocsparse_int* __restrict__ bsr_col_ind,
                       rocsparse_index_base bsr_base)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_int start_local_index = m % block_dim;

    rocsparse_int start = bsr_row_ptr[mb - 1] - bsr_base;
    rocsparse_int end   = bsr_row_ptr[mb] - bsr_base;

    if((start_local_index > 0) && (end - start > 0))
    {
        if((bsr_col_ind[end - 1] - bsr_base) == (mb - 1))
        {
            // then we pad.
            for(rocsparse_int i = start_local_index; i < block_dim; ++i)
            {
                bsr_val[(end - 1) * block_dim * block_dim + i * block_dim + i] = value;
            }
        }
        else
        {
            // search for diagonal block
            for(rocsparse_int index = start; index < end; index++)
            {
                if((bsr_col_ind[index] - bsr_base) == (mb - 1))
                {
                    // then we pad.
                    for(int i = start_local_index; i < block_dim; ++i)
                    {
                        bsr_val[index * block_dim * block_dim + i * block_dim + i] = value;
                    }

                    break;
                }
            }
        }
    }
}

// INSTANTIATE

template struct rocsparse_host<float, int32_t, int32_t, _Float16, _Float16, float>;
template struct rocsparse_host<float, int32_t, int32_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<_Float16, int32_t, int32_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<float,
                               int32_t,
                               int32_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               float>;
template struct rocsparse_host<float,
                               int32_t,
                               int32_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16>;
template struct rocsparse_host<float, int32_t, int32_t, float, float, float>;
template struct rocsparse_host<double, int32_t, int32_t, double, double, double>;
template struct rocsparse_host<rocsparse_float_complex,
                               int32_t,
                               int32_t,
                               rocsparse_float_complex,
                               rocsparse_float_complex,
                               rocsparse_float_complex>;
template struct rocsparse_host<rocsparse_double_complex,
                               int32_t,
                               int32_t,
                               rocsparse_double_complex,
                               rocsparse_double_complex,
                               rocsparse_double_complex>;

template struct rocsparse_host<float, int64_t, int32_t, _Float16, _Float16, float>;
template struct rocsparse_host<float, int64_t, int32_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<_Float16, int64_t, int32_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<float,
                               int64_t,
                               int32_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               float>;
template struct rocsparse_host<float,
                               int64_t,
                               int32_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16>;
template struct rocsparse_host<float, int64_t, int32_t, float, float, float>;
template struct rocsparse_host<double, int64_t, int32_t, double, double, double>;
template struct rocsparse_host<rocsparse_float_complex,
                               int64_t,
                               int32_t,
                               rocsparse_float_complex,
                               rocsparse_float_complex,
                               rocsparse_float_complex>;
template struct rocsparse_host<rocsparse_double_complex,
                               int64_t,
                               int32_t,
                               rocsparse_double_complex,
                               rocsparse_double_complex,
                               rocsparse_double_complex>;

template struct rocsparse_host<float, int64_t, int64_t, _Float16, _Float16, float>;
template struct rocsparse_host<float, int64_t, int64_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<_Float16, int64_t, int64_t, _Float16, _Float16, _Float16>;
template struct rocsparse_host<float,
                               int64_t,
                               int64_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               float>;
template struct rocsparse_host<float,
                               int64_t,
                               int64_t,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16,
                               rocsparse_bfloat16>;
template struct rocsparse_host<float, int64_t, int64_t, float, float, float>;
template struct rocsparse_host<double, int64_t, int64_t, double, double, double>;
template struct rocsparse_host<rocsparse_float_complex,
                               int64_t,
                               int64_t,
                               rocsparse_float_complex,
                               rocsparse_float_complex,
                               rocsparse_float_complex>;
template struct rocsparse_host<rocsparse_double_complex,
                               int64_t,
                               int64_t,
                               rocsparse_double_complex,
                               rocsparse_double_complex,
                               rocsparse_double_complex>;

#define INSTANTIATE_GATHER_SCATTER(ITYPE, TTYPE)                                                 \
    template void host_gthr<ITYPE, TTYPE>(                                                       \
        ITYPE nnz, const TTYPE* y, TTYPE* x_val, const ITYPE* x_ind, rocsparse_index_base base); \
    template void host_sctr<ITYPE, TTYPE>(                                                       \
        ITYPE nnz, const TTYPE* x_val, const ITYPE* x_ind, TTYPE* y, rocsparse_index_base base)

#define INSTANTIATE_T(TYPE)                                                                                   \
    template void             host_gthrz<TYPE>(rocsparse_int nnz,                                             \
                                   TYPE * y,                                                      \
                                   TYPE * x_val,                                                  \
                                   const rocsparse_int* x_ind,                                    \
                                   rocsparse_index_base base);                                    \
    template void             host_bsrsv<TYPE>(rocsparse_operation  trans,                                    \
                                   rocsparse_direction  dir,                                      \
                                   rocsparse_int        mb,                                       \
                                   rocsparse_int        nnzb,                                     \
                                   TYPE                 alpha,                                    \
                                   const rocsparse_int* bsr_row_ptr,                              \
                                   const rocsparse_int* bsr_col_ind,                              \
                                   const TYPE*          bsr_val,                                  \
                                   rocsparse_int        bsr_dim,                                  \
                                   const TYPE*          x,                                        \
                                   TYPE*                y,                                        \
                                   rocsparse_diag_type  diag_type,                                \
                                   rocsparse_fill_mode  fill_mode,                                \
                                   rocsparse_index_base base,                                     \
                                   rocsparse_int*       struct_pivot,                             \
                                   rocsparse_int*       numeric_pivot);                                 \
    template void             host_hybmv<TYPE>(rocsparse_operation  trans,                                    \
                                   rocsparse_int        M,                                        \
                                   rocsparse_int        N,                                        \
                                   TYPE                 alpha,                                    \
                                   rocsparse_int        ell_nnz,                                  \
                                   const rocsparse_int* ell_col_ind,                              \
                                   const TYPE*          ell_val,                                  \
                                   rocsparse_int        ell_width,                                \
                                   rocsparse_int        coo_nnz,                                  \
                                   const rocsparse_int* coo_row_ind,                              \
                                   const rocsparse_int* coo_col_ind,                              \
                                   const TYPE*          coo_val,                                  \
                                   const TYPE*          x,                                        \
                                   TYPE                 beta,                                     \
                                   TYPE*                y,                                        \
                                   rocsparse_index_base base);                                    \
    template void             host_gebsrmv<TYPE>(rocsparse_direction  dir,                                    \
                                     rocsparse_operation  trans,                                  \
                                     rocsparse_int        mb,                                     \
                                     rocsparse_int        nb,                                     \
                                     rocsparse_int        nnzb,                                   \
                                     TYPE                 alpha,                                  \
                                     const rocsparse_int* bsr_row_ptr,                            \
                                     const rocsparse_int* bsr_col_ind,                            \
                                     const TYPE*          bsr_val,                                \
                                     rocsparse_int        row_block_dim,                          \
                                     rocsparse_int        col_block_dim,                          \
                                     const TYPE*          x,                                      \
                                     TYPE                 beta,                                   \
                                     TYPE*                y,                                      \
                                     rocsparse_index_base base);                                  \
    template void             host_bsrxmv<TYPE>(rocsparse_direction  dir,                                     \
                                    rocsparse_operation  trans,                                   \
                                    rocsparse_int        size_of_mask,                            \
                                    rocsparse_int        mb,                                      \
                                    rocsparse_int        nb,                                      \
                                    rocsparse_int        nnzb,                                    \
                                    TYPE                 alpha,                                   \
                                    const rocsparse_int* bsr_mask_ptr,                            \
                                    const rocsparse_int* bsr_row_ptr,                             \
                                    const rocsparse_int* bsr_end_ptr,                             \
                                    const rocsparse_int* bsr_col_ind,                             \
                                    const TYPE*          bsr_val,                                 \
                                    rocsparse_int        bsr_dim,                                 \
                                    const TYPE*          x,                                       \
                                    TYPE                 beta,                                    \
                                    TYPE*                y,                                       \
                                    rocsparse_index_base base);                                   \
    template void             host_bsrpad_value<TYPE>(rocsparse_int m,                                        \
                                          rocsparse_int mb,                                       \
                                          rocsparse_int nnzb,                                     \
                                          rocsparse_int block_dim,                                \
                                          TYPE          value,                                    \
                                          TYPE * bsr_val,                                         \
                                          const rocsparse_int* bsr_row_ptr,                       \
                                          const rocsparse_int* bsr_col_ind,                       \
                                          rocsparse_index_base bsr_base);                         \
    template void             host_gebsrmm<TYPE>(rocsparse_handle          handle,                            \
                                     rocsparse_direction       dir,                               \
                                     rocsparse_operation       trans_A,                           \
                                     rocsparse_operation       trans_B,                           \
                                     rocsparse_int             mb,                                \
                                     rocsparse_int             n,                                 \
                                     rocsparse_int             kb,                                \
                                     rocsparse_int             nnzb,                              \
                                     const TYPE*               alpha,                             \
                                     const rocsparse_mat_descr descr,                             \
                                     const TYPE*               bsr_val,                           \
                                     const rocsparse_int*      bsr_row_ptr,                       \
                                     const rocsparse_int*      bsr_col_ind,                       \
                                     rocsparse_int             row_block_dim,                     \
                                     rocsparse_int             col_block_dim,                     \
                                     const TYPE*               B,                                 \
                                     int64_t                   ldb,                               \
                                     const TYPE*               beta,                              \
                                     TYPE*                     C,                                 \
                                     int64_t                   ldc);                                                \
    template void             host_bsrsm<TYPE>(rocsparse_int        mb,                                       \
                                   rocsparse_int        nrhs,                                     \
                                   rocsparse_int        nnzb,                                     \
                                   rocsparse_direction  dir,                                      \
                                   rocsparse_operation  transA,                                   \
                                   rocsparse_operation  transX,                                   \
                                   TYPE                 alpha,                                    \
                                   const rocsparse_int* bsr_row_ptr,                              \
                                   const rocsparse_int* bsr_col_ind,                              \
                                   const TYPE*          bsr_val,                                  \
                                   rocsparse_int        bsr_dim,                                  \
                                   const TYPE*          B,                                        \
                                   int64_t              ldb,                                      \
                                   TYPE*                X,                                        \
                                   int64_t              ldx,                                      \
                                   rocsparse_diag_type  diag_type,                                \
                                   rocsparse_fill_mode  fill_mode,                                \
                                   rocsparse_index_base base,                                     \
                                   rocsparse_int*       struct_pivot,                             \
                                   rocsparse_int*       numeric_pivot);                                 \
    template void             host_gemmi<TYPE>(rocsparse_int        M,                                        \
                                   rocsparse_int        N,                                        \
                                   rocsparse_operation  transA,                                   \
                                   rocsparse_operation  transB,                                   \
                                   TYPE                 alpha,                                    \
                                   const TYPE*          A,                                        \
                                   int64_t              lda,                                      \
                                   const rocsparse_int* csr_row_ptr,                              \
                                   const rocsparse_int* csr_col_ind,                              \
                                   const TYPE*          csr_val,                                  \
                                   TYPE                 beta,                                     \
                                   TYPE*                C,                                        \
                                   int64_t              ldc,                                      \
                                   rocsparse_index_base base);                                    \
    template void             host_bsrgeam_nnzb<TYPE>(rocsparse_direction  dir,                               \
                                          rocsparse_int        Mb,                                \
                                          rocsparse_int        Nb,                                \
                                          rocsparse_int        block_dim,                         \
                                          const TYPE*          alpha,                             \
                                          const rocsparse_int* bsr_row_ptr_A,                     \
                                          const rocsparse_int* bsr_col_ind_A,                     \
                                          const TYPE*          beta,                              \
                                          const rocsparse_int* bsr_row_ptr_B,                     \
                                          const rocsparse_int* bsr_col_ind_B,                     \
                                          rocsparse_int*       bsr_row_ptr_C,                     \
                                          rocsparse_int*       nnzb_C,                            \
                                          rocsparse_index_base base_A,                            \
                                          rocsparse_index_base base_B,                            \
                                          rocsparse_index_base base_C);                           \
    template void             host_bsrgeam<TYPE>(rocsparse_direction  dir,                                    \
                                     rocsparse_int        Mb,                                     \
                                     rocsparse_int        Nb,                                     \
                                     rocsparse_int        block_dim,                              \
                                     const TYPE*          alpha,                                  \
                                     const rocsparse_int* bsr_row_ptr_A,                          \
                                     const rocsparse_int* bsr_col_ind_A,                          \
                                     const TYPE*          bsr_val_A,                              \
                                     const TYPE*          beta,                                   \
                                     const rocsparse_int* bsr_row_ptr_B,                          \
                                     const rocsparse_int* bsr_col_ind_B,                          \
                                     const TYPE*          bsr_val_B,                              \
                                     const rocsparse_int* bsr_row_ptr_C,                          \
                                     rocsparse_int*       bsr_col_ind_C,                          \
                                     TYPE*                bsr_val_C,                              \
                                     rocsparse_index_base base_A,                                 \
                                     rocsparse_index_base base_B,                                 \
                                     rocsparse_index_base base_C);                                \
    template void             host_bsric0<TYPE>(rocsparse_direction               direction,                  \
                                    rocsparse_int                     Mb,                         \
                                    rocsparse_int                     block_dim,                  \
                                    const std::vector<rocsparse_int>& bsr_row_ptr,                \
                                    const std::vector<rocsparse_int>& bsr_col_ind,                \
                                    std::vector<TYPE>&                bsr_val,                    \
                                    rocsparse_index_base              base,                       \
                                    rocsparse_int*                    struct_pivot,               \
                                    rocsparse_int*                    numeric_pivot);                                \
    template void             host_bsrilu0<TYPE>(rocsparse_direction               dir,                       \
                                     rocsparse_int                     mb,                        \
                                     const std::vector<rocsparse_int>& bsr_row_ptr,               \
                                     const std::vector<rocsparse_int>& bsr_col_ind,               \
                                     std::vector<TYPE>&                bsr_val,                   \
                                     rocsparse_int                     bsr_dim,                   \
                                     rocsparse_index_base              base,                      \
                                     rocsparse_int*                    struct_pivot,              \
                                     rocsparse_int*                    numeric_pivot,             \
                                     bool                              boost,                     \
                                     floating_data_t<TYPE>             boost_tol,                 \
                                     TYPE                              boost_val);                                             \
    template void             host_csric0<TYPE>(rocsparse_int                     M,                          \
                                    const std::vector<rocsparse_int>& csr_row_ptr,                \
                                    const std::vector<rocsparse_int>& csr_col_ind,                \
                                    std::vector<TYPE>&                csr_val,                    \
                                    rocsparse_index_base              base,                       \
                                    rocsparse_int*                    struct_pivot,               \
                                    rocsparse_int*                    numeric_pivot,              \
                                    rocsparse_int*                    singular_pivot,             \
                                    double                            tol);                                                  \
    template void             host_csrilu0<TYPE>(rocsparse_int                     M,                         \
                                     const std::vector<rocsparse_int>& csr_row_ptr,               \
                                     const std::vector<rocsparse_int>& csr_col_ind,               \
                                     std::vector<TYPE>&                csr_val,                   \
                                     rocsparse_index_base              base,                      \
                                     rocsparse_int*                    struct_pivot,              \
                                     rocsparse_int*                    numeric_pivot,             \
                                     rocsparse_int*                    singular_pivot,            \
                                     double                            tol,                       \
                                     bool                              boost,                     \
                                     floating_data_t<TYPE>             boost_tol,                 \
                                     TYPE                              boost_val);                                             \
    template void             host_gtsv_no_pivot<TYPE>(rocsparse_int            m,                            \
                                           rocsparse_int            n,                            \
                                           const std::vector<TYPE>& dl,                           \
                                           const std::vector<TYPE>& d,                            \
                                           const std::vector<TYPE>& du,                           \
                                           std::vector<TYPE>&       B,                            \
                                           rocsparse_int            ldb);                                    \
    template void             host_gtsv_no_pivot_strided_batch<TYPE>(rocsparse_int            m,              \
                                                         const std::vector<TYPE>& dl,             \
                                                         const std::vector<TYPE>& d,              \
                                                         const std::vector<TYPE>& du,             \
                                                         std::vector<TYPE>&       x,              \
                                                         rocsparse_int            batch_count,    \
                                                         rocsparse_int            batch_stride);             \
    template void             host_gtsv_interleaved_batch<TYPE>(rocsparse_gtsv_interleaved_alg algo,          \
                                                    rocsparse_int                  m,             \
                                                    const TYPE*                    dl,            \
                                                    const TYPE*                    d,             \
                                                    const TYPE*                    du,            \
                                                    TYPE*                          x,             \
                                                    rocsparse_int                  batch_count,   \
                                                    rocsparse_int                  batch_stride);                  \
    template void             host_gpsv_interleaved_batch<TYPE>(rocsparse_gpsv_interleaved_alg algo,          \
                                                    rocsparse_int                  m,             \
                                                    TYPE * ds,                                    \
                                                    TYPE * dl,                                    \
                                                    TYPE * d,                                     \
                                                    TYPE * du,                                    \
                                                    TYPE * dw,                                    \
                                                    TYPE * x,                                     \
                                                    rocsparse_int batch_count,                    \
                                                    rocsparse_int batch_stride);                  \
    template rocsparse_status host_nnz<TYPE>(rocsparse_direction dirA,                                        \
                                             rocsparse_int       m,                                           \
                                             rocsparse_int       n,                                           \
                                             const TYPE*         A,                                           \
                                             int64_t             lda,                                         \
                                             rocsparse_int*      nnz_per_row_columns,                         \
                                             rocsparse_int*      nnz_total_dev_host_ptr);                          \
    template void             host_bsr_to_csr<TYPE>(rocsparse_direction               direction,              \
                                        rocsparse_int                     mb,                     \
                                        rocsparse_int                     nb,                     \
                                        rocsparse_int                     nnzb,                   \
                                        const std::vector<TYPE>&          bsr_val,                \
                                        const std::vector<rocsparse_int>& bsr_row_ptr,            \
                                        const std::vector<rocsparse_int>& bsr_col_ind,            \
                                        rocsparse_int                     block_dim,              \
                                        rocsparse_index_base              bsr_base,               \
                                        std::vector<TYPE>&                csr_val,                \
                                        std::vector<rocsparse_int>&       csr_row_ptr,            \
                                        std::vector<rocsparse_int>&       csr_col_ind,            \
                                        rocsparse_index_base              csr_base);                           \
    template void             host_csr_to_bsr<TYPE>(rocsparse_direction               direction,              \
                                        rocsparse_int                     m,                      \
                                        rocsparse_int                     n,                      \
                                        rocsparse_int                     nnz,                    \
                                        const std::vector<TYPE>&          csr_val,                \
                                        const std::vector<rocsparse_int>& csr_row_ptr,            \
                                        const std::vector<rocsparse_int>& csr_col_ind,            \
                                        rocsparse_int                     block_dim,              \
                                        rocsparse_index_base              csr_base,               \
                                        std::vector<TYPE>&                bsr_val,                \
                                        std::vector<rocsparse_int>&       bsr_row_ptr,            \
                                        std::vector<rocsparse_int>&       bsr_col_ind,            \
                                        rocsparse_index_base              bsr_base);                           \
    template void             host_csr_to_gebsr<TYPE>(rocsparse_direction               direction,            \
                                          rocsparse_int                     m,                    \
                                          rocsparse_int                     n,                    \
                                          rocsparse_int                     nnz,                  \
                                          const std::vector<TYPE>&          csr_val,              \
                                          const std::vector<rocsparse_int>& csr_row_ptr,          \
                                          const std::vector<rocsparse_int>& csr_col_ind,          \
                                          rocsparse_int                     row_block_dim,        \
                                          rocsparse_int                     col_block_dim,        \
                                          rocsparse_index_base              csr_base,             \
                                          std::vector<TYPE>&                bsr_val,              \
                                          std::vector<rocsparse_int>&       bsr_row_ptr,          \
                                          std::vector<rocsparse_int>&       bsr_col_ind,          \
                                          rocsparse_index_base              bsr_base);                         \
    template void             host_gebsr_to_gebsc<TYPE>(rocsparse_int                     Mb,                 \
                                            rocsparse_int                     Nb,                 \
                                            rocsparse_int                     nnzb,               \
                                            const std::vector<rocsparse_int>& bsr_row_ptr,        \
                                            const std::vector<rocsparse_int>& bsr_col_ind,        \
                                            const std::vector<TYPE>&          bsr_val,            \
                                            rocsparse_int                     row_block_dim,      \
                                            rocsparse_int                     col_block_dim,      \
                                            std::vector<rocsparse_int>&       bsc_row_ind,        \
                                            std::vector<rocsparse_int>&       bsc_col_ptr,        \
                                            std::vector<TYPE>&                bsc_val,            \
                                            rocsparse_action                  action,             \
                                            rocsparse_index_base              base);                           \
    template void             host_gebsr_to_csr<TYPE>(rocsparse_direction               direction,            \
                                          rocsparse_int                     mb,                   \
                                          rocsparse_int                     nb,                   \
                                          rocsparse_int                     nnzb,                 \
                                          const std::vector<TYPE>&          bsr_val,              \
                                          const std::vector<rocsparse_int>& bsr_row_ptr,          \
                                          const std::vector<rocsparse_int>& bsr_col_ind,          \
                                          rocsparse_int                     row_block_dim,        \
                                          rocsparse_int                     col_block_dim,        \
                                          rocsparse_index_base              bsr_base,             \
                                          std::vector<TYPE>&                csr_val,              \
                                          std::vector<rocsparse_int>&       csr_row_ptr,          \
                                          std::vector<rocsparse_int>&       csr_col_ind,          \
                                          rocsparse_index_base              csr_base);                         \
    template void             host_gebsr_to_gebsr<TYPE>(rocsparse_direction               direction,          \
                                            rocsparse_int                     mb,                 \
                                            rocsparse_int                     nb,                 \
                                            rocsparse_int                     nnzb,               \
                                            const std::vector<TYPE>&          bsr_val_A,          \
                                            const std::vector<rocsparse_int>& bsr_row_ptr_A,      \
                                            const std::vector<rocsparse_int>& bsr_col_ind_A,      \
                                            rocsparse_int                     row_block_dim_A,    \
                                            rocsparse_int                     col_block_dim_A,    \
                                            rocsparse_index_base              base_A,             \
                                            std::vector<TYPE>&                bsr_val_C,          \
                                            std::vector<rocsparse_int>&       bsr_row_ptr_C,      \
                                            std::vector<rocsparse_int>&       bsr_col_ind_C,      \
                                            rocsparse_int                     row_block_dim_C,    \
                                            rocsparse_int                     col_block_dim_C,    \
                                            rocsparse_index_base              base_C);                         \
    template void             host_bsr_to_bsc<TYPE>(rocsparse_int               mb,                           \
                                        rocsparse_int               nb,                           \
                                        rocsparse_int               nnzb,                         \
                                        rocsparse_int               bsr_dim,                      \
                                        const rocsparse_int*        bsr_row_ptr,                  \
                                        const rocsparse_int*        bsr_col_ind,                  \
                                        const TYPE*                 bsr_val,                      \
                                        std::vector<rocsparse_int>& bsc_row_ind,                  \
                                        std::vector<rocsparse_int>& bsc_col_ptr,                  \
                                        std::vector<TYPE>&          bsc_val,                      \
                                        rocsparse_index_base        bsr_base,                     \
                                        rocsparse_index_base        bsc_base);                           \
    template void             host_csr_to_hyb<TYPE>(rocsparse_int                     M,                      \
                                        rocsparse_int                     N,                      \
                                        rocsparse_int                     nnz,                    \
                                        const std::vector<rocsparse_int>& csr_row_ptr,            \
                                        const std::vector<rocsparse_int>& csr_col_ind,            \
                                        const std::vector<TYPE>&          csr_val,                \
                                        std::vector<rocsparse_int>&       ell_col_ind,            \
                                        std::vector<TYPE>&                ell_val,                \
                                        rocsparse_int&                    ell_width,              \
                                        rocsparse_int&                    ell_nnz,                \
                                        std::vector<rocsparse_int>&       coo_row_ind,            \
                                        std::vector<rocsparse_int>&       coo_col_ind,            \
                                        std::vector<TYPE>&                coo_val,                \
                                        rocsparse_int&                    coo_nnz,                \
                                        rocsparse_hyb_partition           part,                   \
                                        rocsparse_index_base              base);                               \
    template void             host_csr_to_csr_compress<TYPE>(rocsparse_int                     M,             \
                                                 rocsparse_int                     N,             \
                                                 rocsparse_int                     nnz,           \
                                                 const std::vector<rocsparse_int>& csr_row_ptr_A, \
                                                 const std::vector<rocsparse_int>& csr_col_ind_A, \
                                                 const std::vector<TYPE>&          csr_val_A,     \
                                                 std::vector<rocsparse_int>&       csr_row_ptr_C, \
                                                 std::vector<rocsparse_int>&       csr_col_ind_C, \
                                                 std::vector<TYPE>&                csr_val_C,     \
                                                 rocsparse_index_base              base,          \
                                                 TYPE                              tol);                                       \
    template void             host_ell_to_csr<TYPE>(rocsparse_int                     M,                      \
                                        rocsparse_int                     N,                      \
                                        const std::vector<rocsparse_int>& ell_col_ind,            \
                                        const std::vector<TYPE>&          ell_val,                \
                                        rocsparse_int                     ell_width,              \
                                        std::vector<rocsparse_int>&       csr_row_ptr,            \
                                        std::vector<rocsparse_int>&       csr_col_ind,            \
                                        std::vector<TYPE>&                csr_val,                \
                                        rocsparse_int&                    csr_nnz,                \
                                        rocsparse_index_base              ell_base,               \
                                        rocsparse_index_base              csr_base);                           \
    template void             host_coosort_by_column<TYPE>(rocsparse_int M,                                   \
                                               rocsparse_int nnz,                                 \
                                               std::vector<rocsparse_int> & coo_row_ind,          \
                                               std::vector<rocsparse_int> & coo_col_ind,          \
                                               std::vector<TYPE> & coo_val);

#define INSTANTIATE_T_REAL_ONLY(TYPE)                                                          \
    template void host_prune_csr_to_csr<TYPE>(rocsparse_int                     M,             \
                                              rocsparse_int                     N,             \
                                              rocsparse_int                     nnz_A,         \
                                              const std::vector<rocsparse_int>& csr_row_ptr_A, \
                                              const std::vector<rocsparse_int>& csr_col_ind_A, \
                                              const std::vector<TYPE>&          csr_val_A,     \
                                              rocsparse_int&                    nnz_C,         \
                                              std::vector<rocsparse_int>&       csr_row_ptr_C, \
                                              std::vector<rocsparse_int>&       csr_col_ind_C, \
                                              std::vector<TYPE>&                csr_val_C,     \
                                              rocsparse_index_base              csr_base_A,    \
                                              rocsparse_index_base              csr_base_C,    \
                                              TYPE                              threshold);                                 \
    template void host_prune_csr_to_csr_by_percentage<TYPE>(                                   \
        rocsparse_int                     M,                                                   \
        rocsparse_int                     N,                                                   \
        rocsparse_int                     nnz_A,                                               \
        const std::vector<rocsparse_int>& csr_row_ptr_A,                                       \
        const std::vector<rocsparse_int>& csr_col_ind_A,                                       \
        const std::vector<TYPE>&          csr_val_A,                                           \
        rocsparse_int&                    nnz_C,                                               \
        std::vector<rocsparse_int>&       csr_row_ptr_C,                                       \
        std::vector<rocsparse_int>&       csr_col_ind_C,                                       \
        std::vector<TYPE>&                csr_val_C,                                           \
        rocsparse_index_base              csr_base_A,                                          \
        rocsparse_index_base              csr_base_C,                                          \
        TYPE                              percentage);                                                                      \
    template void host_prune_dense2csr<TYPE>(rocsparse_int               m,                    \
                                             rocsparse_int               n,                    \
                                             const std::vector<TYPE>&    A,                    \
                                             int64_t                     lda,                  \
                                             rocsparse_index_base        base,                 \
                                             TYPE                        threshold,            \
                                             rocsparse_int&              nnz,                  \
                                             std::vector<TYPE>&          csr_val,              \
                                             std::vector<rocsparse_int>& csr_row_ptr,          \
                                             std::vector<rocsparse_int>& csr_col_ind);         \
    template void host_prune_dense2csr_by_percentage<TYPE>(                                    \
        rocsparse_int               m,                                                         \
        rocsparse_int               n,                                                         \
        const std::vector<TYPE>&    A,                                                         \
        int64_t                     lda,                                                       \
        rocsparse_index_base        base,                                                      \
        TYPE                        percentage,                                                \
        rocsparse_int&              nnz,                                                       \
        std::vector<TYPE>&          csr_val,                                                   \
        std::vector<rocsparse_int>& csr_row_ptr,                                               \
        std::vector<rocsparse_int>& csr_col_ind);

#define INSTANTIATE_IXYT_AXPBY(ITYPE, XTYPE, YTYPE, TTYPE)                           \
    template void host_axpby<TTYPE, ITYPE, XTYPE, YTYPE>(ITYPE                size,  \
                                                         ITYPE                nnz,   \
                                                         TTYPE                alpha, \
                                                         const XTYPE*         x_val, \
                                                         const ITYPE*         x_ind, \
                                                         TTYPE                beta,  \
                                                         YTYPE*               y,     \
                                                         rocsparse_index_base base);

#define INSTANTIATE_COO2DENSE(ITYPE, TTYPE)                                              \
    template void host_coo_to_dense<ITYPE, TTYPE>(ITYPE                     m,           \
                                                  ITYPE                     n,           \
                                                  int64_t                   nnz,         \
                                                  rocsparse_index_base      base,        \
                                                  const std::vector<TTYPE>& coo_val,     \
                                                  const std::vector<ITYPE>& coo_row_ind, \
                                                  const std::vector<ITYPE>& coo_col_ind, \
                                                  std::vector<TTYPE>&       A,           \
                                                  int64_t                   ld,          \
                                                  rocsparse_order           order);
#define INSTANTIATE_DENSE2COO(ITYPE, TTYPE)                                              \
    template void host_dense_to_coo<ITYPE, TTYPE>(ITYPE                     m,           \
                                                  ITYPE                     n,           \
                                                  rocsparse_index_base      base,        \
                                                  const std::vector<TTYPE>& A,           \
                                                  int64_t                   ld,          \
                                                  rocsparse_order           order,       \
                                                  const std::vector<ITYPE>& nnz_per_row, \
                                                  std::vector<TTYPE>&       coo_val,     \
                                                  std::vector<ITYPE>&       coo_row_ind, \
                                                  std::vector<ITYPE>&       coo_col_ind);

#define INSTANTIATE_IT(ITYPE, TTYPE)                                          \
    template void host_gemvi<ITYPE, TTYPE>(ITYPE                M,            \
                                           ITYPE                N,            \
                                           TTYPE                alpha,        \
                                           const TTYPE*         A,            \
                                           int64_t              lda,          \
                                           ITYPE                nnz,          \
                                           const TTYPE*         x_val,        \
                                           const ITYPE*         x_ind,        \
                                           TTYPE                beta,         \
                                           TTYPE*               y,            \
                                           rocsparse_index_base base);        \
    template void host_coosv<ITYPE, TTYPE>(rocsparse_operation  trans,        \
                                           ITYPE                M,            \
                                           int64_t              nnz,          \
                                           TTYPE                alpha,        \
                                           const ITYPE*         coo_row_ind,  \
                                           const ITYPE*         coo_col_ind,  \
                                           const TTYPE*         coo_val,      \
                                           const TTYPE*         x,            \
                                           TTYPE*               y,            \
                                           rocsparse_diag_type  diag_type,    \
                                           rocsparse_fill_mode  fill_mode,    \
                                           rocsparse_index_base base,         \
                                           ITYPE*               struct_pivot, \
                                           ITYPE*               numeric_pivot);             \
    template void host_coosm<ITYPE, TTYPE>(ITYPE                M,            \
                                           ITYPE                nrhs,         \
                                           int64_t              nnz,          \
                                           rocsparse_operation  transA,       \
                                           rocsparse_operation  transB,       \
                                           TTYPE                alpha,        \
                                           const ITYPE*         coo_row_ind,  \
                                           const ITYPE*         coo_col_ind,  \
                                           const TTYPE*         coo_val,      \
                                           TTYPE*               B,            \
                                           int64_t              ldb,          \
                                           rocsparse_order      order_B,      \
                                           rocsparse_diag_type  diag_type,    \
                                           rocsparse_fill_mode  fill_mode,    \
                                           rocsparse_index_base base,         \
                                           ITYPE*               struct_pivot, \
                                           ITYPE*               numeric_pivot);             \
    template void host_roti<ITYPE, TTYPE>(ITYPE nnz,                          \
                                          TTYPE * x_val,                      \
                                          const ITYPE*         x_ind,         \
                                          TTYPE*               y,             \
                                          const TTYPE*         c,             \
                                          const TTYPE*         s,             \
                                          rocsparse_index_base base);

#define INSTANTIATE_IJT(ITYPE, JTYPE, TTYPE)                                                 \
    template void host_csr_to_csc<ITYPE, JTYPE, TTYPE>(JTYPE                M,               \
                                                       JTYPE                N,               \
                                                       ITYPE                nnz,             \
                                                       const ITYPE*         csr_row_ptr,     \
                                                       const JTYPE*         csr_col_ind,     \
                                                       const TTYPE*         csr_val,         \
                                                       std::vector<JTYPE>&  csc_row_ind,     \
                                                       std::vector<ITYPE>&  csc_col_ptr,     \
                                                       std::vector<TTYPE>&  csc_val,         \
                                                       rocsparse_action     action,          \
                                                       rocsparse_index_base base);           \
    template void host_csrsv<ITYPE, JTYPE, TTYPE>(rocsparse_operation  trans,                \
                                                  JTYPE                M,                    \
                                                  ITYPE                nnz,                  \
                                                  TTYPE                alpha,                \
                                                  const ITYPE*         csr_row_ptr,          \
                                                  const JTYPE*         csr_col_ind,          \
                                                  const TTYPE*         csr_val,              \
                                                  const TTYPE*         x,                    \
                                                  int64_t              x_inc,                \
                                                  TTYPE*               y,                    \
                                                  rocsparse_diag_type  diag_type,            \
                                                  rocsparse_fill_mode  fill_mode,            \
                                                  rocsparse_index_base base,                 \
                                                  JTYPE*               struct_pivot,         \
                                                  JTYPE*               numeric_pivot);                     \
    template void host_csrsm<ITYPE, JTYPE, TTYPE>(JTYPE                M,                    \
                                                  JTYPE                nrhs,                 \
                                                  ITYPE                nnz,                  \
                                                  rocsparse_operation  transA,               \
                                                  rocsparse_operation  transB,               \
                                                  TTYPE                alpha,                \
                                                  const ITYPE*         csr_row_ptr,          \
                                                  const JTYPE*         csr_col_ind,          \
                                                  const TTYPE*         csr_val,              \
                                                  TTYPE*               B,                    \
                                                  int64_t              ldb,                  \
                                                  rocsparse_order      order_B,              \
                                                  rocsparse_diag_type  diag_type,            \
                                                  rocsparse_fill_mode  fill_mode,            \
                                                  rocsparse_index_base base,                 \
                                                  JTYPE*               struct_pivot,         \
                                                  JTYPE*               numeric_pivot);                     \
    template void host_csrgeam_nnz<TTYPE, ITYPE, JTYPE>(JTYPE                M,              \
                                                        JTYPE                N,              \
                                                        const TTYPE*         alpha,          \
                                                        const ITYPE*         csr_row_ptr_A,  \
                                                        const JTYPE*         csr_col_ind_A,  \
                                                        const TTYPE*         beta,           \
                                                        const ITYPE*         csr_row_ptr_B,  \
                                                        const JTYPE*         csr_col_ind_B,  \
                                                        ITYPE*               csr_row_ptr_C,  \
                                                        ITYPE*               nnz_C,          \
                                                        rocsparse_index_base base_A,         \
                                                        rocsparse_index_base base_B,         \
                                                        rocsparse_index_base base_C);        \
    template void host_csrgeam<TTYPE, ITYPE, JTYPE>(JTYPE                M,                  \
                                                    JTYPE                N,                  \
                                                    const TTYPE*         alpha,              \
                                                    const ITYPE*         csr_row_ptr_A,      \
                                                    const JTYPE*         csr_col_ind_A,      \
                                                    const TTYPE*         csr_val_A,          \
                                                    const TTYPE*         beta,               \
                                                    const ITYPE*         csr_row_ptr_B,      \
                                                    const JTYPE*         csr_col_ind_B,      \
                                                    const TTYPE*         csr_val_B,          \
                                                    const ITYPE*         csr_row_ptr_C,      \
                                                    JTYPE*               csr_col_ind_C,      \
                                                    TTYPE*               csr_val_C,          \
                                                    rocsparse_index_base base_A,             \
                                                    rocsparse_index_base base_B,             \
                                                    rocsparse_index_base base_C);            \
    template void host_bsrgemm_nnzb<TTYPE, ITYPE, JTYPE>(JTYPE                Mb,            \
                                                         JTYPE                Nb,            \
                                                         JTYPE                Kb,            \
                                                         JTYPE                block_dim,     \
                                                         const TTYPE*         alpha,         \
                                                         const ITYPE*         bsr_row_ptr_A, \
                                                         const JTYPE*         bsr_col_ind_A, \
                                                         const ITYPE*         bsr_row_ptr_B, \
                                                         const JTYPE*         bsr_col_ind_B, \
                                                         const TTYPE*         beta,          \
                                                         const ITYPE*         bsr_row_ptr_D, \
                                                         const JTYPE*         bsr_col_ind_D, \
                                                         ITYPE*               bsr_row_ptr_C, \
                                                         ITYPE*               nnzb_C,        \
                                                         rocsparse_index_base base_A,        \
                                                         rocsparse_index_base base_B,        \
                                                         rocsparse_index_base base_C,        \
                                                         rocsparse_index_base base_D);       \
    template void host_bsrgemm<TTYPE, ITYPE, JTYPE>(rocsparse_direction  dir,                \
                                                    JTYPE                Mb,                 \
                                                    JTYPE                Nb,                 \
                                                    JTYPE                Kb,                 \
                                                    JTYPE                block_dim,          \
                                                    const TTYPE*         alpha,              \
                                                    const ITYPE*         bsr_row_ptr_A,      \
                                                    const JTYPE*         bsr_col_ind_A,      \
                                                    const TTYPE*         bsr_val_A,          \
                                                    const ITYPE*         bsr_row_ptr_B,      \
                                                    const JTYPE*         bsr_col_ind_B,      \
                                                    const TTYPE*         bsr_val_B,          \
                                                    const TTYPE*         beta,               \
                                                    const ITYPE*         bsr_row_ptr_D,      \
                                                    const JTYPE*         bsr_col_ind_D,      \
                                                    const TTYPE*         bsr_val_D,          \
                                                    const ITYPE*         bsr_row_ptr_C,      \
                                                    JTYPE*               bsr_col_ind_C,      \
                                                    TTYPE*               bsr_val_C,          \
                                                    rocsparse_index_base base_A,             \
                                                    rocsparse_index_base base_B,             \
                                                    rocsparse_index_base base_C,             \
                                                    rocsparse_index_base base_D);            \
    template void host_csrgemm_nnz<TTYPE, ITYPE, JTYPE>(JTYPE                M,              \
                                                        JTYPE                N,              \
                                                        JTYPE                K,              \
                                                        const TTYPE*         alpha,          \
                                                        const ITYPE*         csr_row_ptr_A,  \
                                                        const JTYPE*         csr_col_ind_A,  \
                                                        const ITYPE*         csr_row_ptr_B,  \
                                                        const JTYPE*         csr_col_ind_B,  \
                                                        const TTYPE*         beta,           \
                                                        const ITYPE*         csr_row_ptr_D,  \
                                                        const JTYPE*         csr_col_ind_D,  \
                                                        ITYPE*               csr_row_ptr_C,  \
                                                        ITYPE*               nnz_C,          \
                                                        rocsparse_index_base base_A,         \
                                                        rocsparse_index_base base_B,         \
                                                        rocsparse_index_base base_C,         \
                                                        rocsparse_index_base base_D);        \
    template void host_csrgemm<TTYPE, ITYPE, JTYPE>(JTYPE                M,                  \
                                                    JTYPE                N,                  \
                                                    JTYPE                L,                  \
                                                    const TTYPE*         alpha,              \
                                                    const ITYPE*         csr_row_ptr_A,      \
                                                    const JTYPE*         csr_col_ind_A,      \
                                                    const TTYPE*         csr_val_A,          \
                                                    const ITYPE*         csr_row_ptr_B,      \
                                                    const JTYPE*         csr_col_ind_B,      \
                                                    const TTYPE*         csr_val_B,          \
                                                    const TTYPE*         beta,               \
                                                    const ITYPE*         csr_row_ptr_D,      \
                                                    const JTYPE*         csr_col_ind_D,      \
                                                    const TTYPE*         csr_val_D,          \
                                                    const ITYPE*         csr_row_ptr_C,      \
                                                    JTYPE*               csr_col_ind_C,      \
                                                    TTYPE*               csr_val_C,          \
                                                    rocsparse_index_base base_A,             \
                                                    rocsparse_index_base base_B,             \
                                                    rocsparse_index_base base_C,             \
                                                    rocsparse_index_base base_D);

#define INSTANTIATE_IXYT(ITYPE, XTYPE, YTYPE, TTYPE)                                  \
    template void host_doti<ITYPE, XTYPE, YTYPE, TTYPE>(ITYPE                nnz,     \
                                                        const XTYPE*         x_val,   \
                                                        const ITYPE*         x_ind,   \
                                                        const YTYPE*         y,       \
                                                        TTYPE*               result,  \
                                                        rocsparse_index_base base);   \
    template void host_dotci<ITYPE, XTYPE, YTYPE, TTYPE>(ITYPE                nnz,    \
                                                         const XTYPE*         x_val,  \
                                                         const ITYPE*         x_ind,  \
                                                         const YTYPE*         y,      \
                                                         TTYPE*               result, \
                                                         rocsparse_index_base base);

#define INSTANTIATE_CSX2DENSE(DIR, ITYPE, JTYPE, TTYPE)                                          \
    template void host_csx2dense<DIR, TTYPE, ITYPE, JTYPE>(JTYPE                m,               \
                                                           JTYPE                n,               \
                                                           rocsparse_index_base base,            \
                                                           rocsparse_order      order,           \
                                                           const TTYPE*         csx_val,         \
                                                           const ITYPE*         csx_row_col_ptr, \
                                                           const JTYPE*         csx_col_row_ind, \
                                                           TTYPE*               A,               \
                                                           int64_t              ld);

#define INSTANTIATE_DENSE2CSX(DIR, ITYPE, JTYPE, TTYPE)                                              \
    template void host_dense2csx<DIR, TTYPE, ITYPE, JTYPE>(JTYPE                m,                   \
                                                           JTYPE                n,                   \
                                                           rocsparse_index_base base,                \
                                                           const TTYPE*         A,                   \
                                                           int64_t              ld,                  \
                                                           rocsparse_order      order,               \
                                                           const ITYPE*         nnz_per_row_columns, \
                                                           TTYPE*               csx_val,             \
                                                           ITYPE*               csx_row_col_ptr,     \
                                                           JTYPE*               csx_col_row_ind);

#define INSTANTIATE_IJAXYT(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE) \
    template void host_bsrmv(rocsparse_direction  dir,               \
                             rocsparse_operation  trans,             \
                             JTYPE                mb,                \
                             JTYPE                nb,                \
                             ITYPE                nnzb,              \
                             TTYPE                alpha,             \
                             const ITYPE*         bsr_row_ptr,       \
                             const ITYPE*         bsr_end_ptr,       \
                             const JTYPE*         bsr_col_ind,       \
                             const ATYPE*         bsr_val,           \
                             JTYPE                bsr_dim,           \
                             const XTYPE*         x,                 \
                             TTYPE                beta,              \
                             YTYPE*               y,                 \
                             rocsparse_index_base base);             \
    template void host_bsrmv(rocsparse_direction  dir,               \
                             rocsparse_operation  trans,             \
                             JTYPE                mb,                \
                             JTYPE                nb,                \
                             ITYPE                nnzb,              \
                             TTYPE                alpha,             \
                             const ITYPE*         bsr_row_ptr,       \
                             const JTYPE*         bsr_col_ind,       \
                             const ATYPE*         bsr_val,           \
                             JTYPE                bsr_dim,           \
                             const XTYPE*         x,                 \
                             TTYPE                beta,              \
                             YTYPE*               y,                 \
                             rocsparse_index_base base);             \
    template void host_cscmv(rocsparse_operation   trans,            \
                             JTYPE                 M,                \
                             JTYPE                 N,                \
                             ITYPE                 nnz,              \
                             TTYPE                 alpha,            \
                             const ITYPE*          csc_col_ptr,      \
                             const JTYPE*          csc_row_ind,      \
                             const ATYPE*          csc_val,          \
                             const XTYPE*          x,                \
                             TTYPE                 beta,             \
                             YTYPE*                y,                \
                             rocsparse_index_base  base,             \
                             rocsparse_matrix_type matrix_type,      \
                             rocsparse_spmv_alg    algo);               \
    template void host_csrmv(rocsparse_operation   trans,            \
                             JTYPE                 M,                \
                             JTYPE                 N,                \
                             ITYPE                 nnz,              \
                             TTYPE                 alpha,            \
                             const ITYPE*          csr_row_ptr,      \
                             const JTYPE*          csr_col_ind,      \
                             const ATYPE*          csr_val,          \
                             const XTYPE*          x,                \
                             TTYPE                 beta,             \
                             YTYPE*                y,                \
                             rocsparse_index_base  base,             \
                             rocsparse_matrix_type matrix_type,      \
                             rocsparse_spmv_alg    algo,             \
                             bool                  force_conj)

#define INSTANTIATE_IABCT(ITYPE, ATYPE, BTYPE, CTYPE, TTYPE)              \
    template void host_coomm(ITYPE                M,                      \
                             ITYPE                N,                      \
                             ITYPE                K,                      \
                             int64_t              NNZ,                    \
                             rocsparse_operation  transA,                 \
                             rocsparse_operation  transB,                 \
                             TTYPE                alpha,                  \
                             const ITYPE*         coo_row_ind_A,          \
                             const ITYPE*         coo_col_ind_A,          \
                             const ATYPE*         coo_val_A,              \
                             const BTYPE*         dense_B,                \
                             int64_t              ldb,                    \
                             rocsparse_order      order_B,                \
                             TTYPE                beta,                   \
                             CTYPE*               dense_C,                \
                             int64_t              ldc,                    \
                             rocsparse_order      order_C,                \
                             rocsparse_index_base base);                  \
    template void host_coomm_batched(ITYPE                M,              \
                                     ITYPE                N,              \
                                     ITYPE                K,              \
                                     int64_t              NNZ,            \
                                     ITYPE                batch_count_A,  \
                                     int64_t              batch_stride_A, \
                                     rocsparse_operation  transA,         \
                                     rocsparse_operation  transB,         \
                                     TTYPE                alpha,          \
                                     const ITYPE*         coo_row_ind_A,  \
                                     const ITYPE*         coo_col_ind_A,  \
                                     const ATYPE*         coo_val_A,      \
                                     const BTYPE*         dense_B,        \
                                     int64_t              ldb,            \
                                     ITYPE                batch_count_B,  \
                                     int64_t              batch_stride_B, \
                                     rocsparse_order      order_B,        \
                                     TTYPE                beta,           \
                                     CTYPE*               dense_C,        \
                                     int64_t              ldc,            \
                                     ITYPE                batch_count_C,  \
                                     int64_t              batch_stride_C, \
                                     rocsparse_order      order_C,        \
                                     rocsparse_index_base base);

#define INSTANTIATE_IJABCT(ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, TTYPE)                     \
    template void host_csrmm(JTYPE                M,                                     \
                             JTYPE                N,                                     \
                             JTYPE                K,                                     \
                             rocsparse_operation  transA,                                \
                             rocsparse_operation  transB,                                \
                             TTYPE                alpha,                                 \
                             const ITYPE*         csr_row_ptr_A,                         \
                             const JTYPE*         csr_col_ind_A,                         \
                             const ATYPE*         csr_val_A,                             \
                             const BTYPE*         dense_B,                               \
                             int64_t              ldb,                                   \
                             rocsparse_order      order_B,                               \
                             TTYPE                beta,                                  \
                             CTYPE*               dense_C,                               \
                             int64_t              ldc,                                   \
                             rocsparse_order      order_C,                               \
                             rocsparse_index_base base,                                  \
                             bool                 force_conj_A);                                         \
    template void host_cscmm(JTYPE                M,                                     \
                             JTYPE                N,                                     \
                             JTYPE                K,                                     \
                             rocsparse_operation  transA,                                \
                             rocsparse_operation  transB,                                \
                             TTYPE                alpha,                                 \
                             const ITYPE*         csc_col_ptr_A,                         \
                             const JTYPE*         csc_row_ind_A,                         \
                             const ATYPE*         csc_val_A,                             \
                             const BTYPE*         dense_B,                               \
                             int64_t              ldb,                                   \
                             rocsparse_order      order_B,                               \
                             TTYPE                beta,                                  \
                             CTYPE*               dense_C,                               \
                             int64_t              ldc,                                   \
                             rocsparse_order      order_C,                               \
                             rocsparse_index_base base);                                 \
    template void host_bsrmm(rocsparse_handle     handle,                                \
                             rocsparse_direction  dir,                                   \
                             rocsparse_operation  transA,                                \
                             rocsparse_operation  transB,                                \
                             JTYPE                Mb,                                    \
                             JTYPE                N,                                     \
                             JTYPE                Kb,                                    \
                             ITYPE                nnzb,                                  \
                             TTYPE                alpha,                                 \
                             const ATYPE*         bsr_val_A,                             \
                             const ITYPE*         bsr_row_ptr_A,                         \
                             const JTYPE*         bsr_col_ind_A,                         \
                             JTYPE                block_dim,                             \
                             const BTYPE*         dense_B,                               \
                             int64_t              ldb,                                   \
                             rocsparse_order      order_B,                               \
                             TTYPE                beta,                                  \
                             CTYPE*               dense_C,                               \
                             int64_t              ldc,                                   \
                             rocsparse_order      order_C,                               \
                             rocsparse_index_base base);                                 \
    template void host_csrmm_batched(JTYPE                M,                             \
                                     JTYPE                N,                             \
                                     JTYPE                K,                             \
                                     JTYPE                batch_count_A,                 \
                                     int64_t              offsets_batch_stride_A,        \
                                     int64_t              columns_values_batch_stride_A, \
                                     rocsparse_operation  transA,                        \
                                     rocsparse_operation  transB,                        \
                                     TTYPE                alpha,                         \
                                     const ITYPE*         csr_row_ptr_A,                 \
                                     const JTYPE*         csr_col_ind_A,                 \
                                     const ATYPE*         csr_val_A,                     \
                                     const BTYPE*         dense_B,                       \
                                     int64_t              ldb,                           \
                                     JTYPE                batch_count_B,                 \
                                     int64_t              batch_stride_B,                \
                                     rocsparse_order      order_B,                       \
                                     TTYPE                beta,                          \
                                     CTYPE*               dense_C,                       \
                                     int64_t              ldc,                           \
                                     JTYPE                batch_count_C,                 \
                                     int64_t              batch_stride_C,                \
                                     rocsparse_order      order_C,                       \
                                     rocsparse_index_base base,                          \
                                     bool                 force_conj_A);                                 \
    template void host_cscmm_batched(JTYPE                M,                             \
                                     JTYPE                N,                             \
                                     JTYPE                K,                             \
                                     JTYPE                batch_count_A,                 \
                                     int64_t              offsets_batch_stride_A,        \
                                     int64_t              rows_values_batch_stride_A,    \
                                     rocsparse_operation  transA,                        \
                                     rocsparse_operation  transB,                        \
                                     TTYPE                alpha,                         \
                                     const ITYPE*         csc_col_ptr_A,                 \
                                     const JTYPE*         csc_row_ind_A,                 \
                                     const ATYPE*         csc_val_A,                     \
                                     const BTYPE*         dense_B,                       \
                                     int64_t              ldb,                           \
                                     JTYPE                batch_count_B,                 \
                                     int64_t              batch_stride_B,                \
                                     rocsparse_order      order_B,                       \
                                     TTYPE                beta,                          \
                                     CTYPE*               dense_C,                       \
                                     int64_t              ldc,                           \
                                     JTYPE                batch_count_C,                 \
                                     int64_t              batch_stride_C,                \
                                     rocsparse_order      order_C,                       \
                                     rocsparse_index_base base);

#define INSTANTIATE_IAXYT(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)   \
    template void host_coomv(rocsparse_operation  trans,       \
                             ITYPE                M,           \
                             ITYPE                N,           \
                             int64_t              nnz,         \
                             TTYPE                alpha,       \
                             const ITYPE*         coo_row_ind, \
                             const ITYPE*         coo_col_ind, \
                             const ATYPE*         coo_val,     \
                             const XTYPE*         x,           \
                             TTYPE                beta,        \
                             YTYPE*               y,           \
                             rocsparse_index_base base);       \
    template void host_coomv_aos(rocsparse_operation  trans,   \
                                 ITYPE                M,       \
                                 ITYPE                N,       \
                                 int64_t              nnz,     \
                                 TTYPE                alpha,   \
                                 const ITYPE*         coo_ind, \
                                 const ATYPE*         coo_val, \
                                 const XTYPE*         x,       \
                                 TTYPE                beta,    \
                                 YTYPE*               y,       \
                                 rocsparse_index_base base);   \
    template void host_ellmv(rocsparse_operation  trans,       \
                             ITYPE                M,           \
                             ITYPE                N,           \
                             TTYPE                alpha,       \
                             const ITYPE*         ell_col_ind, \
                             const ATYPE*         ell_val,     \
                             ITYPE                ell_width,   \
                             const XTYPE*         x,           \
                             TTYPE                beta,        \
                             YTYPE*               y,           \
                             rocsparse_index_base base);

INSTANTIATE_GATHER_SCATTER(int32_t, int8_t);
INSTANTIATE_GATHER_SCATTER(int32_t, _Float16);
INSTANTIATE_GATHER_SCATTER(int32_t, rocsparse_bfloat16);
INSTANTIATE_GATHER_SCATTER(int32_t, float);
INSTANTIATE_GATHER_SCATTER(int32_t, double);
INSTANTIATE_GATHER_SCATTER(int32_t, rocsparse_float_complex);
INSTANTIATE_GATHER_SCATTER(int32_t, rocsparse_double_complex);
INSTANTIATE_GATHER_SCATTER(int64_t, int8_t);
INSTANTIATE_GATHER_SCATTER(int64_t, _Float16);
INSTANTIATE_GATHER_SCATTER(int64_t, rocsparse_bfloat16);
INSTANTIATE_GATHER_SCATTER(int64_t, float);
INSTANTIATE_GATHER_SCATTER(int64_t, double);
INSTANTIATE_GATHER_SCATTER(int64_t, rocsparse_float_complex);
INSTANTIATE_GATHER_SCATTER(int64_t, rocsparse_double_complex);

INSTANTIATE_T(float);
INSTANTIATE_T(double);
INSTANTIATE_T(rocsparse_float_complex);
INSTANTIATE_T(rocsparse_double_complex);

INSTANTIATE_T_REAL_ONLY(float);
INSTANTIATE_T_REAL_ONLY(double);

INSTANTIATE_IXYT_AXPBY(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_IXYT_AXPBY(int32_t, _Float16, _Float16, float);
INSTANTIATE_IXYT_AXPBY(int32_t, float, float, float);
INSTANTIATE_IXYT_AXPBY(int32_t, double, double, double);
INSTANTIATE_IXYT_AXPBY(int32_t,
                       rocsparse_float_complex,
                       rocsparse_float_complex,
                       rocsparse_float_complex);
INSTANTIATE_IXYT_AXPBY(int32_t,
                       rocsparse_double_complex,
                       rocsparse_double_complex,
                       rocsparse_double_complex);
INSTANTIATE_IXYT_AXPBY(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_IXYT_AXPBY(int64_t, _Float16, _Float16, float);
INSTANTIATE_IXYT_AXPBY(int64_t, float, float, float);
INSTANTIATE_IXYT_AXPBY(int64_t, double, double, double);
INSTANTIATE_IXYT_AXPBY(int64_t,
                       rocsparse_float_complex,
                       rocsparse_float_complex,
                       rocsparse_float_complex);
INSTANTIATE_IXYT_AXPBY(int64_t,
                       rocsparse_double_complex,
                       rocsparse_double_complex,
                       rocsparse_double_complex);

INSTANTIATE_COO2DENSE(int32_t, _Float16);
INSTANTIATE_COO2DENSE(int32_t, rocsparse_bfloat16);
INSTANTIATE_COO2DENSE(int32_t, float);
INSTANTIATE_COO2DENSE(int32_t, double);
INSTANTIATE_COO2DENSE(int32_t, rocsparse_float_complex);
INSTANTIATE_COO2DENSE(int32_t, rocsparse_double_complex);
INSTANTIATE_COO2DENSE(int64_t, _Float16);
INSTANTIATE_COO2DENSE(int64_t, rocsparse_bfloat16);
INSTANTIATE_COO2DENSE(int64_t, float);
INSTANTIATE_COO2DENSE(int64_t, double);
INSTANTIATE_COO2DENSE(int64_t, rocsparse_float_complex);
INSTANTIATE_COO2DENSE(int64_t, rocsparse_double_complex);

INSTANTIATE_DENSE2COO(int32_t, _Float16);
INSTANTIATE_DENSE2COO(int32_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2COO(int32_t, float);
INSTANTIATE_DENSE2COO(int32_t, double);
INSTANTIATE_DENSE2COO(int32_t, rocsparse_float_complex);
INSTANTIATE_DENSE2COO(int32_t, rocsparse_double_complex);
INSTANTIATE_DENSE2COO(int64_t, _Float16);
INSTANTIATE_DENSE2COO(int64_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2COO(int64_t, float);
INSTANTIATE_DENSE2COO(int64_t, double);
INSTANTIATE_DENSE2COO(int64_t, rocsparse_float_complex);
INSTANTIATE_DENSE2COO(int64_t, rocsparse_double_complex);

INSTANTIATE_IT(int32_t, float);
INSTANTIATE_IT(int32_t, double);
INSTANTIATE_IT(int32_t, rocsparse_float_complex);
INSTANTIATE_IT(int32_t, rocsparse_double_complex);
INSTANTIATE_IT(int64_t, float);
INSTANTIATE_IT(int64_t, double);
INSTANTIATE_IT(int64_t, rocsparse_float_complex);
INSTANTIATE_IT(int64_t, rocsparse_double_complex);

INSTANTIATE_IJT(int32_t, int32_t, float);
INSTANTIATE_IJT(int32_t, int32_t, double);
INSTANTIATE_IJT(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_IJT(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_IJT(int64_t, int32_t, float);
INSTANTIATE_IJT(int64_t, int32_t, double);
INSTANTIATE_IJT(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_IJT(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_IJT(int64_t, int64_t, float);
INSTANTIATE_IJT(int64_t, int64_t, double);
INSTANTIATE_IJT(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_IJT(int64_t, int64_t, rocsparse_double_complex);

INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_row, int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, _Float16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, float);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, double);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_CSX2DENSE(rocsparse_direction_column, int64_t, int64_t, rocsparse_double_complex);

INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_row, int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, _Float16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, float);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, double);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_DENSE2CSX(rocsparse_direction_column, int64_t, int64_t, rocsparse_double_complex);

INSTANTIATE_IXYT(int32_t, float, float, float);
INSTANTIATE_IXYT(int64_t, float, float, float);
INSTANTIATE_IXYT(int32_t, double, double, double);
INSTANTIATE_IXYT(int64_t, double, double, double);
INSTANTIATE_IXYT(int32_t,
                 rocsparse_float_complex,
                 rocsparse_float_complex,
                 rocsparse_float_complex);
INSTANTIATE_IXYT(int64_t,
                 rocsparse_float_complex,
                 rocsparse_float_complex,
                 rocsparse_float_complex);
INSTANTIATE_IXYT(int32_t,
                 rocsparse_double_complex,
                 rocsparse_double_complex,
                 rocsparse_double_complex);
INSTANTIATE_IXYT(int64_t,
                 rocsparse_double_complex,
                 rocsparse_double_complex,
                 rocsparse_double_complex);
INSTANTIATE_IXYT(int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_IXYT(int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_IXYT(int32_t, _Float16, _Float16, float);
INSTANTIATE_IXYT(int64_t, _Float16, _Float16, float);
INSTANTIATE_IXYT(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_IXYT(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_IXYT(int32_t, int8_t, int8_t, float);
INSTANTIATE_IXYT(int64_t, int8_t, int8_t, float);

INSTANTIATE_IJAXYT(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJAXYT(int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, int8_t, int8_t, float, float);
INSTANTIATE_IJAXYT(int32_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, _Float16, _Float16, float, float);
INSTANTIATE_IJAXYT(int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJAXYT(int32_t, int32_t, float, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int32_t, float, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int64_t, float, double, double, double);

INSTANTIATE_IJAXYT(int32_t,
                   int32_t,
                   float,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int32_t,
                   float,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int64_t,
                   float,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int32_t,
                   int32_t,
                   double,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int32_t,
                   double,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int64_t,
                   double,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);

INSTANTIATE_IJAXYT(int32_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int64_t,
                   rocsparse_float_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);

INSTANTIATE_IJAXYT(int32_t, int32_t, float, float, float, float);
INSTANTIATE_IJAXYT(int64_t, int32_t, float, float, float, float);
INSTANTIATE_IJAXYT(int64_t, int64_t, float, float, float, float);
INSTANTIATE_IJAXYT(int32_t, int32_t, double, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int32_t, double, double, double, double);
INSTANTIATE_IJAXYT(int64_t, int64_t, double, double, double, double);
INSTANTIATE_IJAXYT(int32_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int64_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJAXYT(int32_t,
                   int32_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int32_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJAXYT(int64_t,
                   int64_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);

INSTANTIATE_IABCT(int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IABCT(int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IABCT(int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IABCT(int64_t, int8_t, int8_t, float, float);
INSTANTIATE_IABCT(int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IABCT(int64_t, _Float16, _Float16, float, float);
INSTANTIATE_IABCT(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IABCT(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IABCT(int32_t, float, float, float, float);
INSTANTIATE_IABCT(int64_t, float, float, float, float);
INSTANTIATE_IABCT(int32_t, double, double, double, double);
INSTANTIATE_IABCT(int64_t, double, double, double, double);
INSTANTIATE_IABCT(int32_t,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_IABCT(int64_t,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_IABCT(int32_t,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_IABCT(int64_t,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

INSTANTIATE_IJABCT(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJABCT(int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJABCT(int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IJABCT(int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJABCT(int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IJABCT(int64_t, int64_t, int8_t, int8_t, float, float);
INSTANTIATE_IJABCT(int32_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IJABCT(int64_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IJABCT(int64_t, int64_t, _Float16, _Float16, float, float);
INSTANTIATE_IJABCT(int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJABCT(int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJABCT(int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IJABCT(int32_t, int32_t, float, float, float, float);
INSTANTIATE_IJABCT(int64_t, int32_t, float, float, float, float);
INSTANTIATE_IJABCT(int64_t, int64_t, float, float, float, float);
INSTANTIATE_IJABCT(int32_t, int32_t, double, double, double, double);
INSTANTIATE_IJABCT(int64_t, int32_t, double, double, double, double);
INSTANTIATE_IJABCT(int64_t, int64_t, double, double, double, double);
INSTANTIATE_IJABCT(int32_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJABCT(int64_t,
                   int32_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJABCT(int64_t,
                   int64_t,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex,
                   rocsparse_float_complex);
INSTANTIATE_IJABCT(int32_t,
                   int32_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJABCT(int64_t,
                   int32_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);
INSTANTIATE_IJABCT(int64_t,
                   int64_t,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex,
                   rocsparse_double_complex);

INSTANTIATE_IAXYT(int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IAXYT(int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_IAXYT(int32_t, int8_t, int8_t, float, float);
INSTANTIATE_IAXYT(int64_t, int8_t, int8_t, float, float);
INSTANTIATE_IAXYT(int32_t, _Float16, _Float16, float, float);
INSTANTIATE_IAXYT(int64_t, _Float16, _Float16, float, float);
INSTANTIATE_IAXYT(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IAXYT(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_IAXYT(
    int32_t, float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_IAXYT(
    int64_t, float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_IAXYT(int32_t, float, double, double, double);
INSTANTIATE_IAXYT(int64_t, float, double, double, double);
INSTANTIATE_IAXYT(
    int32_t, double, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE_IAXYT(
    int64_t, double, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE_IAXYT(int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_IAXYT(int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_IAXYT(int32_t, float, float, float, float);
INSTANTIATE_IAXYT(int64_t, float, float, float, float);
INSTANTIATE_IAXYT(int32_t, double, double, double, double);
INSTANTIATE_IAXYT(int64_t, double, double, double, double);
INSTANTIATE_IAXYT(int32_t,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_IAXYT(int64_t,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_IAXYT(int32_t,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_IAXYT(int64_t,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

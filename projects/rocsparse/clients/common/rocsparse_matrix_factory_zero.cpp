/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_init.hpp"

#include "rocsparse_matrix_factory_zero.hpp"

#include "rocsparse_clients_routine_trace.hpp"

template <typename T, typename I, typename J>
rocsparse_matrix_factory_zero<T, I, J>::rocsparse_matrix_factory_zero(){};

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_zero<T, I, J>::init_csr(std::vector<I>&        csr_row_ptr,
                                                      std::vector<J>&        csr_col_ind,
                                                      std::vector<T>&        csr_val,
                                                      J&                     M,
                                                      J&                     N,
                                                      I&                     nnz,
                                                      rocsparse_index_base   base,
                                                      rocsparse_matrix_type  matrix_type,
                                                      rocsparse_fill_mode    uplo,
                                                      rocsparse_storage_mode storage)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    csr_row_ptr.resize((M > 0) ? (M + 1) : 0, static_cast<I>(base));
    csr_col_ind.resize(0);
    csr_val.resize(0);

    nnz = 0;
}

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_zero<T, I, J>::init_gebsr(std::vector<I>&        bsr_row_ptr,
                                                        std::vector<J>&        bsr_col_ind,
                                                        std::vector<T>&        bsr_val,
                                                        rocsparse_direction    dirb,
                                                        J&                     Mb,
                                                        J&                     Nb,
                                                        I&                     nnzb,
                                                        J&                     row_block_dim,
                                                        J&                     col_block_dim,
                                                        rocsparse_index_base   base,
                                                        rocsparse_matrix_type  matrix_type,
                                                        rocsparse_fill_mode    uplo,
                                                        rocsparse_storage_mode storage)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    bsr_row_ptr.resize((Mb > 0) ? (Mb + 1) : 0, static_cast<I>(base));
    bsr_col_ind.resize(0);
    bsr_val.resize(0);

    nnzb = 0;
}

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_zero<T, I, J>::init_coo(std::vector<I>&        coo_row_ind,
                                                      std::vector<I>&        coo_col_ind,
                                                      std::vector<T>&        coo_val,
                                                      I&                     M,
                                                      I&                     N,
                                                      int64_t&               nnz,
                                                      rocsparse_index_base   base,
                                                      rocsparse_matrix_type  matrix_type,
                                                      rocsparse_fill_mode    uplo,
                                                      rocsparse_storage_mode storage)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    coo_row_ind.resize(0);
    coo_col_ind.resize(0);
    coo_val.resize(0);

    nnz = 0;
}

template struct rocsparse_matrix_factory_zero<int8_t, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<int8_t, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<int8_t, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<_Float16, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<_Float16, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<_Float16, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<rocsparse_bfloat16, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_bfloat16, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_bfloat16, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<float, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<float, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<float, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<double, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<double, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<double, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<rocsparse_float_complex, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_float_complex, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_float_complex, int64_t, int64_t>;

template struct rocsparse_matrix_factory_zero<rocsparse_double_complex, int32_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_double_complex, int64_t, int32_t>;
template struct rocsparse_matrix_factory_zero<rocsparse_double_complex, int64_t, int64_t>;

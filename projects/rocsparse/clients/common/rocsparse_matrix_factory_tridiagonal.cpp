/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_matrix_utils.hpp"

#include "rocsparse_matrix_factory_tridiagonal.hpp"

#include "rocsparse_clients_routine_trace.hpp"

template <typename T, typename I, typename J>
rocsparse_matrix_factory_tridiagonal<T, I, J>::rocsparse_matrix_factory_tridiagonal(J l, J u)
    : m_l(l)
    , m_u(u){};

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_tridiagonal<T, I, J>::init_csr(std::vector<I>&        csr_row_ptr,
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

    switch(matrix_type)
    {
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        std::vector<I> ptr;
        std::vector<J> ind;
        std::vector<T> val;

        rocsparse_init_csr_tridiagonal(ptr, ind, val, M, N, nnz, base, this->m_l, this->m_u);

        rocsparse_matrix_utils::host_csrtri(ptr.data(),
                                            ind.data(),
                                            val.data(),
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            M,
                                            N,
                                            nnz,
                                            base,
                                            uplo);
        break;
    }
    case rocsparse_matrix_type_general:
    {
        rocsparse_init_csr_tridiagonal(
            csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, this->m_l, this->m_u);
        break;
    }
    }

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_csrunsort<T, I, J>(
            csr_row_ptr.data(), csr_col_ind.data(), M, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }
}

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_tridiagonal<T, I, J>::init_coo(std::vector<I>&        coo_row_ind,
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

    switch(matrix_type)
    {
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        std::vector<I> row_ind;
        std::vector<I> col_ind;
        std::vector<T> val;

        rocsparse_init_coo_tridiagonal(
            row_ind, col_ind, val, M, N, nnz, base, (I)this->m_l, (I)this->m_u);

        rocsparse_matrix_utils::host_cootri(row_ind.data(),
                                            col_ind.data(),
                                            val.data(),
                                            coo_row_ind,
                                            coo_col_ind,
                                            coo_val,
                                            M,
                                            N,
                                            nnz,
                                            base,
                                            uplo);
        break;
    }
    case rocsparse_matrix_type_general:
    {
        rocsparse_init_coo_tridiagonal(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, (I)this->m_l, (I)this->m_u);
        break;
    }
    }

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_coounsort<T, I>(
            coo_row_ind.data(), coo_col_ind.data(), M, nnz, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }
}

template <typename T, typename I, typename J>
void rocsparse_matrix_factory_tridiagonal<T, I, J>::init_gebsr(std::vector<I>&        bsr_row_ptr,
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

    rocsparse_init_gebsr_tridiagonal(bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_val,
                                     Mb,
                                     Nb,
                                     nnzb,
                                     row_block_dim,
                                     col_block_dim,
                                     base,
                                     this->m_l,
                                     this->m_u);

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_gebsrunsort<T, I, J>(
            bsr_row_ptr.data(), bsr_col_ind.data(), Mb, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }
}

template struct rocsparse_matrix_factory_tridiagonal<int8_t, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<int8_t, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<int8_t, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<_Float16, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<_Float16, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<_Float16, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<rocsparse_bfloat16, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_bfloat16, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_bfloat16, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<float, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<float, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<float, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<double, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<double, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<double, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<rocsparse_float_complex, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_float_complex, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_float_complex, int64_t, int64_t>;

template struct rocsparse_matrix_factory_tridiagonal<rocsparse_double_complex, int32_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_double_complex, int64_t, int32_t>;
template struct rocsparse_matrix_factory_tridiagonal<rocsparse_double_complex, int64_t, int64_t>;

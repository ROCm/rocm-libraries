/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "display.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "testing_matrix_type_traits.hpp"
#include "utility.hpp"

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
struct testing_spmv_dispatch_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_csr, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csr, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_csr(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_csrmv<T, I, J, A, X, Y>(trans,
                                     hA.m,
                                     hA.n,
                                     hA.nnz,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base,
                                     matrix_type,
                                     alg,
                                     false);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return csrmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_csc, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csc, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_csc(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {

        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_cscmv<T, I, J, A, X, Y>(trans,
                                     hA.m,
                                     hA.n,
                                     hA.nnz,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base,
                                     matrix_type,
                                     alg);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return cscmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR BSR FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_bsr, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_bsr, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        J block_dim = matrix_factory.m_arg.block_dim;
        matrix_factory.init_gebsr(hA, m, n, block_dim, block_dim, base);
        m *= block_dim;
        n *= block_dim;
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.mb * dA.row_block_dim,
                            display_key_t::N,
                            dA.nb * dA.col_block_dim,
                            display_key_t::nnz,
                            dA.nnzb * dA.row_block_dim * dA.col_block_dim,
                            display_key_t::bdim,
                            dA.row_block_dim,
                            display_key_t::bdir,
                            rocsparse_direction2string(dA.block_direction),
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_bsrmv<T, I, J, A, X, Y>(hA.block_direction,
                                     trans,
                                     hA.mb,
                                     hA.nb,
                                     hA.nnzb,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hA.row_block_dim,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return bsrmv_gbyte_count<A, X, T>(hA.mb, hA.nb, hA.nnzb, hA.row_block_dim, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(
            hA.mb * hA.row_block_dim, hA.nnzb * hA.row_block_dim * hA.col_block_dim, nonzero_beta);
    }
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_coo, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo, I, I, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_coo(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_coomv<T, I, A, X, Y>(trans,
                                  hA.m,
                                  hA.n,
                                  hA.nnz,
                                  *h_alpha,
                                  hA.row_ind,
                                  hA.col_ind,
                                  hA.val,
                                  hx,
                                  *h_beta,
                                  hy,
                                  hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return coomv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_coo_aos, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_coo_aos(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_coomv_aos<T, I, A, X, Y>(
            trans, hA.m, hA.n, hA.nnz, *h_alpha, hA.ind, hA.val, hx, *h_beta, hy, hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return coomv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_ell, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_ell, I, I, A>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_ell(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_ellmv<T, I, A, X, Y>(
            trans, hA.m, hA.n, *h_alpha, hA.ind, hA.val, hA.width, hx, *h_beta, hy, hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return ellmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

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

#pragma once
#ifndef ROCSPARSE_MATRIX_UTILS_HPP
#define ROCSPARSE_MATRIX_UTILS_HPP

#include "rocsparse.hpp"
#include "rocsparse_matrix.hpp"

static inline const float* get_boost_tol(const float* tol)
{
    return tol;
}

static inline const double* get_boost_tol(const double* tol)
{
    return tol;
}

static inline const float* get_boost_tol(const rocsparse_float_complex* tol)
{
    return reinterpret_cast<const float*>(tol);
}

static inline const double* get_boost_tol(const rocsparse_double_complex* tol)
{
    return reinterpret_cast<const double*>(tol);
}

//
// @brief Utility methods for matrices..
//
struct rocsparse_matrix_utils
{

    //
    // @brief Initialize a host dense matrix with random values.
    // @param[in] that Fill \p that matrix.
    // @param[in] use_exact If \p use_exact is true, the matrix will be initialized with exact values.
    //
    template <typename T>
    static void init(host_dense_matrix<T>& that, bool use_exact = false)
    {
        switch(that.order)
        {
        case rocsparse_order_column:
        {
            rocsparse_init<T>(that, that.m, that.n, that.ld, use_exact);
            break;
        }

        case rocsparse_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocsparse_order_column.
            // If this poses a problem, we need to refactor rocsparse_init.
            //
            rocsparse_init<T>(that, that.n, that.m, that.ld, use_exact);
            break;
        }
        }
    }

    //
    // @brief Initialize a host dense matrix with random integer values.
    // @param[in] that Fill \p that matrix.
    //
    template <typename T>
    static void init_exact(host_dense_matrix<T>& that)
    {
        switch(that.order)
        {
        case rocsparse_order_column:
        {
            rocsparse_init_exact<T>(that, that.m, that.n, that.ld);
            break;
        }

        case rocsparse_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocsparse_order_column.
            // If this poses a problem, we need to refactor rocsparse_init_exact.
            //
            rocsparse_init_exact<T>(that, that.n, that.m, that.ld);
            break;
        }
        }
    }

    //
    // @brief Compress a \p device_csr_matrix
    // @param[out] result Define a \p device_csr_matrix resulting from the compression of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void compress(device_csr_matrix<T>&       result,
                         const device_csr_matrix<T>& that,
                         rocsparse_index_base        base)
    {
        result.define(that.m, that.n, 0, base);
        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
        rocsparse_mat_descr descr;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_mat_descr(&descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_set_mat_index_base(descr, that.base));
        T                            tol   = static_cast<T>(0);
        rocsparse_int                nnz_c = 0;
        device_vector<rocsparse_int> dnnz_per_row(that.m);
        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_nnz_compress<T>(
            handle, that.m, descr, that.val, that.ptr, dnnz_per_row, &nnz_c, tol));
        result.define(that.m, that.n, nnz_c, base);
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                                  that.m,
                                                                  that.n,
                                                                  descr,
                                                                  that.val,
                                                                  that.ptr,
                                                                  that.ind,
                                                                  that.nnz,
                                                                  dnnz_per_row,
                                                                  result.val,
                                                                  result.ptr,
                                                                  result.ind,
                                                                  tol));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_mat_descr(descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    //
    // @brief Convert a \p device_csr_matrix into a \p device_gebsr_matrix.
    // @param[out] result Define a \p device_csr_matrix resulting from the conversion of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               row_block_dim,
                        rocsparse_int               col_block_dim,
                        rocsparse_index_base        base,
                        rocsparse_storage_mode      storage,
                        device_gebsr_matrix<T>&     result)
    {
        // Currently this routine only works on sorted input and output matrices
        if(storage == rocsparse_storage_mode_unsorted)
        {
            std::cerr << "Error: rocsparse_matrix_utils::convert only works with sorted matrices"
                      << std::endl;
            throw rocsparse_status_internal_error;
        }

        rocsparse_int mb = (that.m + row_block_dim - 1) / row_block_dim;
        rocsparse_int nb = (that.n + col_block_dim - 1) / col_block_dim;

        rocsparse_int nnzb = 0;

        result.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));

        rocsparse_mat_descr that_descr;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_mat_descr(&that_descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_set_mat_index_base(that_descr, that.base));

        rocsparse_mat_descr result_descr;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_mat_descr(&result_descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_set_mat_index_base(result_descr, base));

        // Convert CSR to GEBSR
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2gebsr_buffer_size<T>(handle,
                                                                       result.block_direction,
                                                                       that.m,
                                                                       that.n,
                                                                       that_descr,
                                                                       that.val,
                                                                       that.ptr,
                                                                       that.ind,
                                                                       result.row_block_dim,
                                                                       result.col_block_dim,
                                                                       &buffer_size));

        int* buffer = NULL;
        CHECK_HIP_THROW_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                            result.block_direction,
                                                            that.m,
                                                            that.n,
                                                            that_descr,
                                                            that.ptr,
                                                            that.ind,
                                                            result_descr,
                                                            result.ptr,
                                                            result.row_block_dim,
                                                            result.col_block_dim,
                                                            &nnzb,
                                                            buffer));

        result.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                           result.block_direction,
                                                           that.m,
                                                           that.n,
                                                           that_descr,
                                                           that.val,
                                                           that.ptr,
                                                           that.ind,
                                                           result_descr,
                                                           result.val,
                                                           result.ptr,
                                                           result.ind,
                                                           result.row_block_dim,
                                                           result.col_block_dim,
                                                           buffer));

        CHECK_HIP_THROW_ERROR(rocsparse_hipFree(buffer));

        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_mat_descr(result_descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_mat_descr(that_descr));
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    //
    // @brief Convert a \p device_csr_matrix into a \p device_gebsr_matrix.
    // @param[out] result Define a \p device_gebsr_matrix resulting from the conversion of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               block_dim,
                        rocsparse_index_base        base,
                        rocsparse_storage_mode      storage,
                        device_gebsr_matrix<T>&     c)
    {
        // Currently this routine only works on sorted input and output matrices
        if(storage == rocsparse_storage_mode_unsorted)
        {
            std::cerr << "Error: rocsparse_matrix_utils::convert only works with sorted matrices"
                      << std::endl;
            throw rocsparse_status_internal_error;
        }

        rocsparse_int mb   = (that.m + block_dim - 1) / block_dim;
        rocsparse_int nb   = (that.n + block_dim - 1) / block_dim;
        rocsparse_int nnzb = 0;

        c.define(dirb, mb, nb, nnzb, block_dim, block_dim, base);

        {

            rocsparse_handle handle;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));

            rocsparse_mat_descr that_descr;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_mat_descr(&that_descr));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_set_mat_index_base(that_descr, that.base));

            rocsparse_mat_descr c_descr;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_mat_descr(&c_descr));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_set_mat_index_base(c_descr, base));
            // Convert sample CSR matrix to bsr
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                              dirb,
                                                              that.m,
                                                              that.n,
                                                              that_descr,
                                                              that.ptr,
                                                              that.ind,
                                                              block_dim,
                                                              c_descr,
                                                              c.ptr,
                                                              &nnzb));

            c.define(dirb, mb, nb, nnzb, block_dim, block_dim, base);

            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csr2bsr<T>(handle,
                                                             dirb,
                                                             that.m,
                                                             that.n,
                                                             that_descr,
                                                             that.val,
                                                             that.ptr,
                                                             that.ind,
                                                             block_dim,
                                                             c_descr,
                                                             c.val,
                                                             c.ptr,
                                                             c.ind));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_mat_descr(c_descr));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_mat_descr(that_descr));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
        }
    }

    typedef enum
    {
        bsrilu0_analysis = 1,
        bsrilu0_solve    = 2,
        bsrilu0_all      = 3
    } bsrilu0_step;

    template <typename T>
    static void bsrilu0(rocsparse_mat_descr       that_descr,
                        device_gebsr_matrix<T>&   that,
                        rocsparse_mat_info        info,
                        rocsparse_analysis_policy analysis,
                        rocsparse_solve_policy    solve,
                        int                       boost,
                        T                         h_boost_val,
                        T                         h_boost_tol,
                        size_t*                   p_buffer_size,
                        void*                     buffer,
                        bsrilu0_step              step = bsrilu0_all)
    {

        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_THROW_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                                         that.block_direction,
                                                                         that.mb,
                                                                         that.nnzb,
                                                                         that_descr,
                                                                         that.val,
                                                                         that.ptr,
                                                                         that.ind,
                                                                         that.row_block_dim,
                                                                         info,
                                                                         p_buffer_size));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));

            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case bsrilu0_all:
        case bsrilu0_analysis:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                                      that.block_direction,
                                                                      that.mb,
                                                                      that.nnzb,
                                                                      that_descr,
                                                                      that.val,
                                                                      that.ptr,
                                                                      that.ind,
                                                                      that.row_block_dim,
                                                                      info,
                                                                      analysis,
                                                                      solve,
                                                                      buffer));
            if(step != bsrilu0_all)
            {
                break;
            }
        }

        case bsrilu0_solve:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsrilu0_numeric_boost<T>(
                handle, info, boost, get_boost_tol(&h_boost_tol), &h_boost_val));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsrilu0<T>(handle,
                                                             that.block_direction,
                                                             that.mb,
                                                             that.nnzb,
                                                             that_descr,
                                                             that.val,
                                                             that.ptr,
                                                             that.ind,
                                                             that.row_block_dim,
                                                             info,
                                                             solve,
                                                             buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        bsric0_analysis = 1,
        bsric0_solve    = 2,
        bsric0_all      = 3
    } bsric0_step;

    template <typename T>
    static void bsric0(rocsparse_mat_descr       that_descr,
                       device_gebsr_matrix<T>&   that,
                       rocsparse_mat_info        info,
                       rocsparse_analysis_policy analysis,
                       rocsparse_solve_policy    solve,
                       size_t*                   p_buffer_size,
                       void*                     buffer,
                       bsric0_step               step = bsric0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_THROW_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsric0_buffer_size<T>(handle,
                                                                        that.block_direction,
                                                                        that.mb,
                                                                        that.nnzb,
                                                                        that_descr,
                                                                        that.val,
                                                                        that.ptr,
                                                                        that.ind,
                                                                        that.row_block_dim,
                                                                        info,
                                                                        p_buffer_size));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case bsric0_all:
        case bsric0_analysis:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                                     that.block_direction,
                                                                     that.mb,
                                                                     that.nnzb,
                                                                     that_descr,
                                                                     that.val,
                                                                     that.ptr,
                                                                     that.ind,
                                                                     that.row_block_dim,
                                                                     info,
                                                                     analysis,
                                                                     solve,
                                                                     buffer));
            if(step != bsric0_all)
            {
                break;
            }
        }

        case bsric0_solve:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_bsric0<T>(handle,
                                                            that.block_direction,
                                                            that.mb,
                                                            that.nnzb,
                                                            that_descr,
                                                            that.val,
                                                            that.ptr,
                                                            that.ind,
                                                            that.row_block_dim,
                                                            info,
                                                            solve,
                                                            buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        csric0_analysis = 1,
        csric0_solve    = 2,
        csric0_all      = 3
    } csric0_step;

    template <typename T>
    static void csric0(rocsparse_mat_descr       that_descr,
                       device_csr_matrix<T>&     that,
                       rocsparse_mat_info        info,
                       rocsparse_analysis_policy analysis,
                       rocsparse_solve_policy    solve,
                       size_t*                   p_buffer_size,
                       void*                     buffer,
                       csric0_step               step = csric0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_THROW_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csric0_buffer_size<T>(handle,
                                                                        that.m,
                                                                        that.nnz,
                                                                        that_descr,
                                                                        that.val,
                                                                        that.ptr,
                                                                        that.ind,
                                                                        info,
                                                                        p_buffer_size));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case csric0_all:
        case csric0_analysis:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                                     that.m,
                                                                     that.nnz,
                                                                     that_descr,
                                                                     that.val,
                                                                     that.ptr,
                                                                     that.ind,
                                                                     info,
                                                                     analysis,
                                                                     solve,
                                                                     buffer));
            if(step != csric0_all)
            {
                break;
            }
        }

        case csric0_solve:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csric0<T>(handle,
                                                            that.m,
                                                            that.nnz,
                                                            that_descr,
                                                            that.val,
                                                            that.ptr,
                                                            that.ind,
                                                            info,
                                                            solve,
                                                            buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            rocsparse_int hsolve_pivot_neg[1];
            {
                auto st = rocsparse_csric0_singular_pivot(handle, info, hsolve_pivot_neg);
                EXPECT_ROCSPARSE_STATUS(st, rocsparse_status_success);
            }
            break;
        }
        }
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        csrilu0_analysis = 1,
        csrilu0_solve    = 2,
        csrilu0_all      = 3
    } csrilu0_step;

    template <typename T>
    static void csrilu0(rocsparse_mat_descr       that_descr,
                        device_csr_matrix<T>&     that,
                        rocsparse_mat_info        info,
                        rocsparse_analysis_policy analysis,
                        rocsparse_solve_policy    solve,
                        int                       boost,
                        T                         h_boost_val,
                        T                         h_boost_tol,
                        size_t*                   p_buffer_size,
                        void*                     buffer,
                        csrilu0_step              step = csrilu0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_THROW_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csrilu0_buffer_size<T>(handle,
                                                                         that.m,
                                                                         that.nnz,
                                                                         that_descr,
                                                                         that.val,
                                                                         that.ptr,
                                                                         that.ind,
                                                                         info,
                                                                         p_buffer_size));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_THROW_ERROR(
            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case csrilu0_all:
        case csrilu0_analysis:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csrilu0_analysis<T>(handle,
                                                                      that.m,
                                                                      that.nnz,
                                                                      that_descr,
                                                                      that.val,
                                                                      that.ptr,
                                                                      that.ind,
                                                                      info,
                                                                      analysis,
                                                                      solve,
                                                                      buffer));
            if(step != csrilu0_all)
            {
                break;
            }
        }

        case csrilu0_solve:
        {
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csrilu0_numeric_boost<T>(
                handle, info, boost, get_boost_tol(&h_boost_tol), &h_boost_val));
            CHECK_ROCSPARSE_THROW_ERROR(rocsparse_csrilu0<T>(handle,
                                                             that.m,
                                                             that.nnz,
                                                             that_descr,
                                                             that.val,
                                                             that.ptr,
                                                             that.ind,
                                                             info,
                                                             solve,
                                                             buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_THROW_ERROR(rocsparse_destroy_handle(handle));
    }

    // Extract lower or upper part of input matrix to form new triangular output matrix
    template <typename T,
              typename I,
              typename J,
              template <typename...>
              class VEC1,
              template <typename...>
              class VEC2,
              template <typename...>
              class VEC3>
    static void host_csrtri(const I*             ptr,
                            const J*             ind,
                            const T*             val,
                            VEC1<I>&             csr_row_ptr,
                            VEC2<J>&             csr_col_ind,
                            VEC3<T>&             csr_val,
                            J                    M,
                            J                    N,
                            I&                   nnz,
                            rocsparse_index_base base,
                            rocsparse_fill_mode  uplo)
    {
        if(M != N)
        {
            std::cerr << "error: host_csrtri only accepts square matrices" << std::endl;
            exit(1);
            return;
        }

        nnz = 0;
        if(uplo == rocsparse_fill_mode_lower)
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base <= i)
                    {
                        nnz++;
                    }
                }
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base >= i)
                    {
                        nnz++;
                    }
                }
            }
        }

        csr_row_ptr.resize(M + 1, 0);
        csr_col_ind.resize(nnz, 0);
        csr_val.resize(nnz, static_cast<T>(0));

        I index        = 0;
        csr_row_ptr[0] = base;

        if(uplo == rocsparse_fill_mode_lower)
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base <= i)
                    {
                        csr_col_ind[index] = ind[j];
                        csr_val[index]     = val[j];
                        index++;
                    }
                }

                csr_row_ptr[i + 1] = index + base;
            }
        }
        else
        {
            for(J i = 0; i < M; i++)
            {
                I start = ptr[i] - base;
                I end   = ptr[i + 1] - base;

                for(I j = start; j < end; j++)
                {
                    if(ind[j] - base >= i)
                    {
                        csr_col_ind[index] = ind[j];
                        csr_val[index]     = val[j];
                        index++;
                    }
                }

                csr_row_ptr[i + 1] = index + base;
            }
        }
    }

    template <typename T,
              typename I,
              template <typename...>
              class VEC1,
              template <typename...>
              class VEC2,
              template <typename...>
              class VEC3>
    static void host_cootri(const I*             row_ind,
                            const I*             col_ind,
                            const T*             val,
                            VEC1<I>&             coo_row_ind,
                            VEC2<I>&             coo_col_ind,
                            VEC3<T>&             coo_val,
                            I                    M,
                            I                    N,
                            int64_t&             nnz,
                            rocsparse_index_base base,
                            rocsparse_fill_mode  uplo)
    {
        if(M != N)
        {
            std::cerr << "error: host_cootri only accepts square matrices" << std::endl;
            exit(1);
            return;
        }

        int64_t old_nnz = nnz;
        int64_t new_nnz = 0;
        if(uplo == rocsparse_fill_mode_lower)
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base <= i)
                    {
                        new_nnz++;
                    }

                    index++;
                }
            }
        }
        else
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base >= i)
                    {
                        new_nnz++;
                    }

                    index++;
                }
            }
        }

        coo_row_ind.resize(new_nnz, 0);
        coo_col_ind.resize(new_nnz, 0);
        coo_val.resize(new_nnz, static_cast<T>(0));

        nnz = 0;
        if(uplo == rocsparse_fill_mode_lower)
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < old_nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base <= i)
                    {
                        coo_row_ind[nnz] = row_ind[index];
                        coo_col_ind[nnz] = col_ind[index];
                        coo_val[nnz]     = val[index];
                        nnz++;
                    }

                    index++;
                }
            }
        }
        else
        {
            int64_t index = 0;
            for(I i = 0; i < M; i++)
            {
                while(index < old_nnz && row_ind[index] - base == i)
                {
                    if(col_ind[index] - base >= i)
                    {
                        coo_row_ind[nnz] = row_ind[index];
                        coo_col_ind[nnz] = col_ind[index];
                        coo_val[nnz]     = val[index];
                        nnz++;
                    }

                    index++;
                }
            }
        }
    }

    // Shuffle matrix columns so that the matrix is unsorted
    template <typename T, typename I, typename J>
    static void host_csrunsort(const I* csr_row_ptr, J* csr_col_ind, J M, rocsparse_index_base base)
    {
        std::random_device rd;
        std::mt19937       generator(rd());

        for(J i = 0; i < M; i++)
        {
            I start = csr_row_ptr[i] - base;
            I end   = csr_row_ptr[i + 1] - base;

            if(start < end)
            {
                std::shuffle(&csr_col_ind[start], &csr_col_ind[end], generator);
            }
        }
    }

    // Shuffle matrix columns so that the matrix is unsorted
    template <typename T, typename I>
    static void host_coounsort(
        const I* coo_row_ind, I* coo_col_ind, I M, int64_t nnz, rocsparse_index_base base)
    {
        std::random_device rd;
        std::mt19937       generator(rd());

        int64_t index = 0;

        for(I i = 0; i < M; i++)
        {
            int64_t start = index;
            while(index < nnz && coo_row_ind[index] - base == i)
            {
                index++;
            }
            int64_t end = index;

            if(start < end)
            {
                std::shuffle(&coo_col_ind[start], &coo_col_ind[end], generator);
            }
        }
    }

    // Shuffle matrix columns so that the matrix is unsorted
    template <typename T, typename I, typename J>
    static void
        host_gebsrunsort(const I* bsr_row_ptr, J* bsr_col_ind, J Mb, rocsparse_index_base base)
    {
        host_csrunsort<T, I, J>(bsr_row_ptr, bsr_col_ind, Mb, base);
    }

    template <typename T, typename I, typename J>
    static rocsparse_status host_csrsym(const host_csr_matrix<T, I, J>& A,
                                        host_csr_matrix<T, I, J>&       symA)
    {
        const auto M    = A.m;
        const auto NNZ  = A.nnz;
        const auto base = A.base;

        if(M != A.n)
        {
            return rocsparse_status_invalid_value;
        }

        //
        // Compute transpose.
        //
        host_csr_matrix<T, I, J> trA(M, M, NNZ, base);
        {
            for(J i = 0; i <= M; ++i)
            {
                trA.ptr[i] = static_cast<I>(0);
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j = A.ind[k] - base;
                    trA.ptr[j + 1] += 1;
                }
            }

            for(J i = 2; i <= M; ++i)
            {
                trA.ptr[i] += trA.ptr[i - 1];
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j           = A.ind[k] - base;
                    trA.ind[trA.ptr[j]] = i;
                    trA.val[trA.ptr[j]] = A.val[k];
                    ++trA.ptr[j];
                }
            }

            for(J i = M; i > 0; --i)
            {
                trA.ptr[i] = trA.ptr[i - 1];
            }
            trA.ptr[0] = 0;

            if(rocsparse_index_base_one == base)
            {
                for(J i = 0; i <= M; ++i)
                {
                    trA.ptr[i] += base;
                }

                for(I i = 0; i < NNZ; ++i)
                {
                    trA.ind[i] += base;
                }
            }
        }
        //
        // Compute transpose done.
        //

        //
        // Compute (A + A^T) / 2
        //
        host_dense_vector<J> blank(M);
        for(size_t i = 0; i < blank.size(); i++)
        {
            blank[i] = 0;
        }
        host_dense_vector<J> select(M);

        symA.define(M, M, 0, base);

        for(J i = 0; i <= M; ++i)
        {
            symA.ptr[i] = 0;
        }

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - trA.base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Reset blank.
            //
            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            symA.ptr[i + 1] = select_n;
        }

        for(J i = 2; i <= M; ++i)
        {
            symA.ptr[i] += symA.ptr[i - 1];
        }

        symA.define(M, M, symA.ptr[M], base);

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            std::sort(select.data(), select.data() + select_n);

            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            for(J k = 0; k < select_n; ++k)
            {
                symA.ind[symA.ptr[i] + k] = select[k];
            }
        }

        if(rocsparse_index_base_one == base)
        {
            for(J i = 0; i <= M; ++i)
            {
                symA.ptr[i] += base;
            }

            for(I i = 0; i < symA.nnz; ++i)
            {
                symA.ind[i] += base;
            }
        }

        return rocsparse_status_success;
    }
};

#endif // ROCSPARSE_MATRIX_UTILS_HPP

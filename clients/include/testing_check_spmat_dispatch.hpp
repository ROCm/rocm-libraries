/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "auto_testing_bad_arg.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_check_spmat_dispatch_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_check_spmat_dispatch_traits<rocsparse_format_csr, I, J, T>
{
    using host_sparse_matrix   = host_csr_matrix<T, I, J>;
    using device_sparse_matrix = device_csr_matrix<T, I, J>;

    static void sparse_initialization(rocsparse_matrix_factory<T, I, J>& matrix_factory,
                                      host_sparse_matrix&                hA,
                                      J                                  m,
                                      J                                  n,
                                      rocsparse_index_base               base,
                                      J                                  block_dim)
    {
        matrix_factory.init_csr(hA, m, n, base);
    }

    static void display_info(const Arguments& arg, host_sparse_matrix& hA, double gpu_time_used)
    {
        double gbyte_count = check_matrix_csr_gbyte_count<T>(hA.m, hA.nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::N,
                            hA.n,
                            display_key_t::nnz,
                            hA.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename T>
struct testing_check_spmat_dispatch_traits<rocsparse_format_csc, I, J, T>
{
    using host_sparse_matrix   = host_csc_matrix<T, I, J>;
    using device_sparse_matrix = device_csc_matrix<T, I, J>;

    static void sparse_initialization(rocsparse_matrix_factory<T, I, J>& matrix_factory,
                                      host_sparse_matrix&                hA,
                                      J                                  m,
                                      J                                  n,
                                      rocsparse_index_base               base,
                                      J                                  block_dim)
    {
        matrix_factory.init_csc(hA, m, n, base);
    }

    static void display_info(const Arguments& arg, host_sparse_matrix& hA, double gpu_time_used)
    {
        double gbyte_count = check_matrix_csc_gbyte_count<T>(hA.n, hA.nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::N,
                            hA.n,
                            display_key_t::nnz,
                            hA.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_check_spmat_dispatch_traits<rocsparse_format_coo, I, I, T>
{
    using host_sparse_matrix   = host_coo_matrix<T, I>;
    using device_sparse_matrix = device_coo_matrix<T, I>;

    static void sparse_initialization(rocsparse_matrix_factory<T, I, I>& matrix_factory,
                                      host_sparse_matrix&                hA,
                                      I                                  m,
                                      I                                  n,
                                      rocsparse_index_base               base,
                                      I                                  block_dim)
    {
        matrix_factory.init_coo(hA, m, n, base);
    }

    static void display_info(const Arguments& arg, host_sparse_matrix& hA, double gpu_time_used)
    {
        double gbyte_count = check_matrix_coo_gbyte_count<T>(hA.nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::N,
                            hA.n,
                            display_key_t::nnz,
                            hA.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_check_spmat_dispatch_traits<rocsparse_format_ell, I, I, T>
{
    using host_sparse_matrix   = host_ell_matrix<T, I>;
    using device_sparse_matrix = device_ell_matrix<T, I>;

    static void sparse_initialization(rocsparse_matrix_factory<T, I, I>& matrix_factory,
                                      host_sparse_matrix&                hA,
                                      I                                  m,
                                      I                                  n,
                                      rocsparse_index_base               base,
                                      I                                  block_dim)
    {
        matrix_factory.init_ell(hA, m, n, base);
    }

    static void display_info(const Arguments& arg, host_sparse_matrix& hA, double gpu_time_used)
    {
        double gbyte_count = check_matrix_ell_gbyte_count<T>(hA.nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::N,
                            hA.n,
                            display_key_t::nnz,
                            hA.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
};

//
// TRAITS FOR BSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_check_spmat_dispatch_traits<rocsparse_format_bsr, I, J, T>
{
    using host_sparse_matrix   = host_gebsr_matrix<T, I, J>;
    using device_sparse_matrix = device_gebsr_matrix<T, I, J>;

    static void sparse_initialization(rocsparse_matrix_factory<T, I, J>& matrix_factory,
                                      host_sparse_matrix&                hA,
                                      J                                  m,
                                      J                                  n,
                                      rocsparse_index_base               base,
                                      J                                  block_dim)
    {
        matrix_factory.init_gebsr(hA, m, n, block_dim, block_dim, base);
    }

    static void display_info(const Arguments& arg, host_sparse_matrix& hA, double gpu_time_used)
    {
        double gbyte_count
            = check_matrix_gebsr_gbyte_count<T>(hA.mb, hA.nnzb, hA.row_block_dim, hA.col_block_dim);
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.mb * hA.row_block_dim,
                            display_key_t::N,
                            hA.nb * hA.row_block_dim,
                            display_key_t::Mb,
                            hA.mb,
                            display_key_t::Nb,
                            hA.nb,
                            display_key_t::bdim,
                            hA.row_block_dim,
                            display_key_t::nnzb,
                            hA.nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
};

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_check_spmat_dispatch
{
private:
    using traits = testing_check_spmat_dispatch_traits<FORMAT, I, J, T>;

    using host_sparse_matrix   = typename traits::host_sparse_matrix;
    using device_sparse_matrix = typename traits::device_sparse_matrix;

public:
    static void testing_check_spmat(const Arguments& arg)
    {
        J                      m           = arg.M;
        J                      n           = arg.N;
        rocsparse_index_base   base        = arg.baseA;
        J                      block_dim   = arg.block_dim;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;

        if(block_dim > 1)
        {
            m = (m + block_dim - 1) / block_dim;
            n = (n + block_dim - 1) / block_dim;
        }

        // Create rocsparse handle
        rocsparse_local_handle handle;

        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

        // Allocate host memory for CSR matrix
        host_sparse_matrix hA;

        // Generate (or load from file) CSR matrix
        traits::sparse_initialization(matrix_factory, hA, m, n, base, block_dim);

        // CSR matrix on device
        device_sparse_matrix dA(hA);

        // Create descriptor
        rocsparse_local_spmat A(dA);

        // Set Attributes
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    A, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);

        // Allocate buffer
        size_t                buffer_size;
        rocsparse_data_status data_status;
        CHECK_ROCSPARSE_ERROR(rocsparse_check_spmat(handle,
                                                    A,
                                                    &data_status,
                                                    rocsparse_check_spmat_stage_buffer_size,
                                                    &buffer_size,
                                                    nullptr));

        void* dbuffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        if(arg.unit_check)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_check_spmat(handle,
                                                        A,
                                                        &data_status,
                                                        rocsparse_check_spmat_stage_compute,
                                                        &buffer_size,
                                                        dbuffer));
            CHECK_ROCSPARSE_DATA_ERROR(data_status);
        }

        if(arg.timing)
        {
            const double gpu_time_used
                = rocsparse_clients::run_benchmark(arg,
                                                   rocsparse_check_spmat,
                                                   handle,
                                                   A,
                                                   &data_status,
                                                   rocsparse_check_spmat_stage_compute,
                                                   nullptr,
                                                   dbuffer);

            traits::display_info(arg, hA, gpu_time_used);
        }

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    }
};

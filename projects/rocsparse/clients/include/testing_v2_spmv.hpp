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

#include "auto_testing_bad_arg.hpp"
#include "rocsparse_matrix_statistics.hpp"
#include "testing_spmv_dispatch_traits.hpp"

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
struct testing_v2_spmv_dispatch
{
private:
    using traits = testing_spmv_dispatch_traits<FORMAT, I, J, A, X, Y, T>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

public:
    static void testing_v2_spmv_bad_arg(const Arguments& arg)
    {
        rocsparse_local_handle  local_handle;
        rocsparse_handle        handle  = local_handle;
        rocsparse_spmv_descr    descr   = (rocsparse_spmv_descr)0x4;
        rocsparse_spmat_descr   mat     = (rocsparse_spmat_descr)0x4;
        const void*             alpha   = (const void*)0x4;
        const void*             beta    = (const void*)0x4;
        rocsparse_v2_spmv_stage stage   = rocsparse_v2_spmv_stage_analysis;
        void*                   buffer  = (void*)0x4;
        rocsparse_dnvec_descr   x       = (rocsparse_dnvec_descr)0x4;
        rocsparse_dnvec_descr   y       = (rocsparse_dnvec_descr)0x4;
        rocsparse_error*        p_error = nullptr;

        {
            size_t* buffer_size_in_bytes = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE handle, descr, mat, x, y, stage, buffer_size_in_bytes, p_error

            static constexpr int nex     = 1;
            static const int     ex[nex] = {7};
            select_bad_arg_analysis(rocsparse_v2_spmv_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);
#undef PARAMS_BUFFER_SIZE
        }

        {
            const size_t buffer_size_in_bytes = 10;

#define PARAMS handle, descr, alpha, mat, x, beta, y, stage, buffer_size_in_bytes, buffer, p_error

            static constexpr int nex     = 2;
            static const int     ex[nex] = {8, 10};
            select_bad_arg_analysis(rocsparse_v2_spmv, nex, ex, PARAMS);
#undef PARAMS
        }
    }

    static void testing_v2_spmv(const Arguments& arg)
    {
        J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_datatype     ttype       = get_datatype<T>();

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

#define PARAMS(alpha_, A_, x_, beta_, y_, stage) \
    handle, trans, alpha_, A_, x_, beta_, y_, ttype, alg, stage, &buffer_size, dbuffer, nullptr

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<A> hA;
        {
            int dev;
            CHECK_HIP_ERROR(hipGetDevice(&dev));

            hipDeviceProp_t prop;
            CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

            const bool has_datafile = rocsparse_arguments_has_datafile(arg);
            bool       to_int       = false;
            to_int |= (prop.warpSize == 32);
            to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);
            to_int |= (trans != rocsparse_operation_none && has_datafile);
            to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<A, I, J> matrix_factory(
                arg, arg.unit_check ? to_int : false, full_rank);
            traits::sparse_initialization(matrix_factory, hA, M, N, base);
        }

        if((matrix_type == rocsparse_matrix_type_symmetric && M != N)
           || (matrix_type == rocsparse_matrix_type_triangular && M != N))
        {
            return;
        }

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx((trans == rocsparse_operation_none) ? N : M, 1);
        rocsparse_matrix_utils::init_exact(hx);
        device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy((trans == rocsparse_operation_none) ? M : N, 1);
        rocsparse_matrix_utils::init_exact(hy);
        device_dense_matrix<Y> dy(hy);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),

            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Run buffer size
        rocsparse_spmv_descr spmv_descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
            handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
            handle, spmv_descr, rocsparse_spmv_input_operation, &trans, sizeof(trans), nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_compute_datatype,
                                                       &ttype,
                                                       sizeof(ttype),
                                                       nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_scalar_datatype,
                                                       &ttype,
                                                       sizeof(ttype),
                                                       nullptr));

        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                            spmv_descr,
                                                            matA,
                                                            x,
                                                            y,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            &buffer_size,
                                                            nullptr));

        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        // Run analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                spmv_descr,
                                                h_alpha,
                                                matA,
                                                x,
                                                h_beta,
                                                y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                buffer_size,
                                                dbuffer,
                                                nullptr));

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        dbuffer = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                            spmv_descr,
                                                            matA,
                                                            x,
                                                            y,
                                                            rocsparse_v2_spmv_stage_compute,
                                                            &buffer_size,
                                                            nullptr));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        if(arg.unit_check)
        {
            // Run solve
            CHECK_ROCSPARSE_ERROR(testing::rocsparse_v2_spmv(handle,
                                                             spmv_descr,
                                                             h_alpha,
                                                             matA,
                                                             x,
                                                             h_beta,
                                                             y,
                                                             rocsparse_v2_spmv_stage_compute,
                                                             buffer_size,
                                                             dbuffer,
                                                             nullptr));

            host_dense_matrix<Y> hy_copy(hy);
            traits::host_calculation(trans, h_alpha, hA, hx, h_beta, hy, alg, matrix_type);

            hy.near_check(dy);

            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y_pointer_mode_host", dy);
            }

            dy.transfer_from(hy_copy);
            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            CHECK_ROCSPARSE_ERROR(testing::rocsparse_v2_spmv(handle,
                                                             spmv_descr,
                                                             d_alpha,
                                                             matA,
                                                             x,
                                                             d_beta,
                                                             y,
                                                             rocsparse_v2_spmv_stage_compute,
                                                             buffer_size,
                                                             dbuffer,
                                                             nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            hy.near_check(dy);
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y_pointer_mode_device", dy);
            }
        }

        if(arg.timing)
        {
            const double gpu_time_used
                = rocsparse_clients::run_benchmark(arg,
                                                   rocsparse_v2_spmv,
                                                   handle,
                                                   spmv_descr,
                                                   h_alpha,
                                                   matA,
                                                   x,
                                                   h_beta,
                                                   y,
                                                   rocsparse_v2_spmv_stage_compute,
                                                   buffer_size,
                                                   dbuffer,
                                                   nullptr);

            const double gflop_count = traits::gflop_count(hA, *h_beta != static_cast<T>(0));
            const double gbyte_count = traits::byte_count(hA, *h_beta != static_cast<T>(0));

            const double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
            const double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

            if(arg.sparsity_pattern_statistics)
            {
                int64_t min_nnz_row;
                int64_t median_nnz_row;
                int64_t max_nnz_row;
                rocsparse_matrix_statistics::get_nnz_per_row(
                    dA, min_nnz_row, median_nnz_row, max_nnz_row);

                int64_t min_nnz_col;
                int64_t median_nnz_col;
                int64_t max_nnz_col;
                rocsparse_matrix_statistics::get_nnz_per_column(
                    dA, min_nnz_col, median_nnz_col, max_nnz_col);
                traits::display_info(arg,
                                     display_key_t::trans_A,
                                     rocsparse_operation2string(trans),
                                     dA,
                                     display_key_t::min_nnz_per_row,
                                     min_nnz_row,
                                     display_key_t::max_nnz_per_row,
                                     max_nnz_row,
                                     display_key_t::median_nnz_per_row,
                                     median_nnz_row,
                                     display_key_t::min_nnz_per_col,
                                     min_nnz_col,
                                     display_key_t::max_nnz_per_col,
                                     max_nnz_col,
                                     display_key_t::median_nnz_per_col,
                                     median_nnz_col,
                                     display_key_t::alpha,
                                     *h_alpha,
                                     display_key_t::beta,
                                     *h_beta,
                                     display_key_t::algorithm,
                                     rocsparse_spmvalg2string(alg),
                                     display_key_t::gflops,
                                     gpu_gflops,
                                     display_key_t::bandwidth,
                                     gpu_gbyte,
                                     display_key_t::time_ms,
                                     get_gpu_time_msec(gpu_time_used));
            }
            else
            {
                traits::display_info(arg,
                                     display_key_t::trans_A,
                                     rocsparse_operation2string(trans),
                                     dA,
                                     display_key_t::alpha,
                                     *h_alpha,
                                     display_key_t::beta,
                                     *h_beta,
                                     display_key_t::algorithm,
                                     rocsparse_spmvalg2string(alg),
                                     display_key_t::gflops,
                                     gpu_gflops,
                                     display_key_t::bandwidth,
                                     gpu_gbyte,
                                     display_key_t::time_ms,
                                     get_gpu_time_msec(gpu_time_used));
            }
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmv_descr(spmv_descr));
        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    }

    static void testing_v2_spmv_extra_bad_arg(const Arguments& arg)
    {
        rocsparse_spmv_descr descr = (rocsparse_spmv_descr)0x4;

        // Test bad arguments for rocsparse_spmv_clear_extra
        static constexpr int nex_clear           = 3;
        static const int     ex_clear[nex_clear] = {0, 1, 2}; // handle, descr, p_error
        select_bad_arg_analysis(rocsparse_spmv_clear_extra,
                                nex_clear,
                                ex_clear,
                                (rocsparse_handle)0x4,
                                descr,
                                (rocsparse_error*)0x4);
    }

    static void testing_v2_spmv_extra(const Arguments& arg)
    {
        J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_datatype     ttype       = get_datatype<T>();

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        T gamma_val = static_cast<T>(0.5);

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());
        host_scalar<T> h_gamma(static_cast<T>(gamma_val));

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);
        device_scalar<T> d_gamma(h_gamma);

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<A> hA;
        {
            int dev;
            CHECK_HIP_ERROR(hipGetDevice(&dev));

            hipDeviceProp_t prop;
            CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

            const bool has_datafile = rocsparse_arguments_has_datafile(arg);
            bool       to_int       = false;
            to_int |= (prop.warpSize == 32);
            to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);
            to_int |= (trans != rocsparse_operation_none && has_datafile);
            to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<A, I, J> matrix_factory(
                arg, arg.unit_check ? to_int : false, full_rank);
            traits::sparse_initialization(matrix_factory, hA, M, N, base);
        }

        if((matrix_type == rocsparse_matrix_type_symmetric && M != N)
           || (matrix_type == rocsparse_matrix_type_triangular && M != N))
        {
            return;
        }

        if(M == 0 || N == 0)
        {
            return;
        }

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx((trans == rocsparse_operation_none) ? N : M, 1);
        rocsparse_matrix_utils::init_exact(hx);
        device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy((trans == rocsparse_operation_none) ? M : N, 1);
        rocsparse_matrix_utils::init_exact(hy);
        device_dense_matrix<Y> dy(hy);

        // Create extra vector
        host_dense_matrix<X> h_extra((trans == rocsparse_operation_none) ? M : N, 1);
        rocsparse_matrix_utils::init_exact(h_extra);
        device_dense_matrix<X> d_extra(h_extra);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);
        rocsparse_local_dnvec extra(d_extra);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Run buffer size
        rocsparse_spmv_descr spmv_descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
            handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
            handle, spmv_descr, rocsparse_spmv_input_operation, &trans, sizeof(trans), nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_compute_datatype,
                                                       &ttype,
                                                       sizeof(ttype),
                                                       nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_scalar_datatype,
                                                       &ttype,
                                                       sizeof(ttype),
                                                       nullptr));

        // Create gamma dnvec with host values
        rocsparse_dnvec_descr gamma_vec;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_dnvec_descr(&gamma_vec, 1, h_gamma, ttype));

        rocsparse_const_dnvec_descr z_vecs[1] = {extra};

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmv_set_extra(handle, spmv_descr, 1, gamma_vec, z_vecs, nullptr));

        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                            spmv_descr,
                                                            matA,
                                                            x,
                                                            y,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            &buffer_size,
                                                            nullptr));

        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        // Run analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                spmv_descr,
                                                h_alpha,
                                                matA,
                                                x,
                                                h_beta,
                                                y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                buffer_size,
                                                dbuffer,
                                                nullptr));

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        dbuffer = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                            spmv_descr,
                                                            matA,
                                                            x,
                                                            y,
                                                            rocsparse_v2_spmv_stage_compute,
                                                            &buffer_size,
                                                            nullptr));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        if(arg.unit_check)
        {
            // Run solve
            CHECK_ROCSPARSE_ERROR(testing::rocsparse_v2_spmv(handle,
                                                             spmv_descr,
                                                             h_alpha,
                                                             matA,
                                                             x,
                                                             h_beta,
                                                             y,
                                                             rocsparse_v2_spmv_stage_compute,
                                                             buffer_size,
                                                             dbuffer,
                                                             nullptr));

            host_dense_matrix<Y> hy_copy(hy);
            traits::host_calculation(trans, h_alpha, hA, hx, h_beta, hy, alg, matrix_type);

            const size_t hy_size = (trans == rocsparse_operation_none) ? M : N;

            for(size_t i = 0; i < hy_size; ++i)
            {
                hy[i] += gamma_val * h_extra[i];
            }

            hy.near_check(dy);

            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y_pointer_mode_host_extra", dy);
            }

            dy.transfer_from(hy_copy);
        }

        // Clear extra vector after computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv_clear_extra(handle, spmv_descr, nullptr));

        if(arg.timing)
        {
            const double gpu_time_used
                = rocsparse_clients::run_benchmark(arg,
                                                   rocsparse_v2_spmv,
                                                   handle,
                                                   spmv_descr,
                                                   h_alpha,
                                                   matA,
                                                   x,
                                                   h_beta,
                                                   y,
                                                   rocsparse_v2_spmv_stage_compute,
                                                   buffer_size,
                                                   dbuffer,
                                                   nullptr);

            const double gflop_count = traits::gflop_count(hA, *h_beta != static_cast<T>(0));
            const double gbyte_count = traits::byte_count(hA, *h_beta != static_cast<T>(0));

            const double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
            const double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

            if(arg.sparsity_pattern_statistics)
            {
                int64_t min_nnz_row;
                int64_t median_nnz_row;
                int64_t max_nnz_row;
                rocsparse_matrix_statistics::get_nnz_per_row(
                    dA, min_nnz_row, median_nnz_row, max_nnz_row);

                int64_t min_nnz_col;
                int64_t median_nnz_col;
                int64_t max_nnz_col;
                rocsparse_matrix_statistics::get_nnz_per_column(
                    dA, min_nnz_col, median_nnz_col, max_nnz_col);
                traits::display_info(arg,
                                     display_key_t::trans_A,
                                     rocsparse_operation2string(trans),
                                     dA,
                                     display_key_t::min_nnz_per_row,
                                     min_nnz_row,
                                     display_key_t::max_nnz_per_row,
                                     max_nnz_row,
                                     display_key_t::median_nnz_per_row,
                                     median_nnz_row,
                                     display_key_t::min_nnz_per_col,
                                     min_nnz_col,
                                     display_key_t::max_nnz_per_col,
                                     max_nnz_col,
                                     display_key_t::median_nnz_per_col,
                                     median_nnz_col,
                                     display_key_t::alpha,
                                     *h_alpha,
                                     display_key_t::beta,
                                     *h_beta,
                                     display_key_t::algorithm,
                                     rocsparse_spmvalg2string(alg),
                                     display_key_t::gflops,
                                     gpu_gflops,
                                     display_key_t::bandwidth,
                                     gpu_gbyte,
                                     display_key_t::time_ms,
                                     get_gpu_time_msec(gpu_time_used));
            }
            else
            {
                traits::display_info(arg,
                                     display_key_t::trans_A,
                                     rocsparse_operation2string(trans),
                                     dA,
                                     display_key_t::alpha,
                                     *h_alpha,
                                     display_key_t::beta,
                                     *h_beta,
                                     display_key_t::algorithm,
                                     rocsparse_spmvalg2string(alg),
                                     display_key_t::gflops,
                                     gpu_gflops,
                                     display_key_t::bandwidth,
                                     gpu_gbyte,
                                     display_key_t::time_ms,
                                     get_gpu_time_msec(gpu_time_used));
            }
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(gamma_vec));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmv_descr(spmv_descr));
        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

        // Test enable/disable extra functionality
        if(matrix_type == rocsparse_matrix_type_general && arg.unit_check)
        {
            // Create a new spmv descriptor for enable/disable testing
            rocsparse_spmv_descr spmv_descr;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

            // Set mandatory inputs
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
                handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_operation,
                                                           &trans,
                                                           sizeof(trans),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_scalar_datatype,
                                                           &ttype,
                                                           sizeof(ttype),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_compute_datatype,
                                                           &ttype,
                                                           sizeof(ttype),
                                                           nullptr));

            // Set extra vectors
            T                     gamma_val = static_cast<T>(1.0);
            rocsparse_dnvec_descr gamma_vec;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_dnvec_descr(&gamma_vec, 1, &gamma_val, ttype));
            rocsparse_const_dnvec_descr z_vecs[1] = {extra};

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spmv_set_extra(handle, spmv_descr, 1, gamma_vec, z_vecs, nullptr));

            // Run analysis
            size_t buffer_size = 0;
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                                spmv_descr,
                                                                matA,
                                                                x,
                                                                y,
                                                                rocsparse_v2_spmv_stage_analysis,
                                                                &buffer_size,
                                                                nullptr));

            void* dbuffer = nullptr;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                    spmv_descr,
                                                    h_alpha,
                                                    matA,
                                                    x,
                                                    h_beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_analysis,
                                                    buffer_size,
                                                    dbuffer,
                                                    nullptr));

            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

            // Test compute stage with extras enabled (default state)
            buffer_size = 0;
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                                spmv_descr,
                                                                matA,
                                                                x,
                                                                y,
                                                                rocsparse_v2_spmv_stage_compute,
                                                                &buffer_size,
                                                                nullptr));
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

            // Store original y values for comparison
            host_dense_matrix<Y> hy_enabled(hy);
            hy_enabled.transfer_from(dy);

            // Run with extras enabled
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                    spmv_descr,
                                                    h_alpha,
                                                    matA,
                                                    x,
                                                    h_beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_compute,
                                                    buffer_size,
                                                    dbuffer,
                                                    nullptr));

            host_dense_matrix<Y> hy_with_extra(hy);
            hy_with_extra.transfer_from(dy);

            // Reset y values
            dy.transfer_from(hy_enabled);

            // Disable extras and run again
            int32_t disable_extra = 0;

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_enable_extra,
                                                           &disable_extra,
                                                           sizeof(disable_extra),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                    spmv_descr,
                                                    h_alpha,
                                                    matA,
                                                    x,
                                                    h_beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_compute,
                                                    buffer_size,
                                                    dbuffer,
                                                    nullptr));

            host_dense_matrix<Y> hy_without_extra(hy);
            hy_without_extra.transfer_from(dy);

            // Verify results are different (extras disabled should not include z vector)
            const size_t hy_size           = (trans == rocsparse_operation_none) ? M : N;
            int          results_different = 0;
            for(size_t i = 0; i < hy_size; ++i)
            {
                if(hy_with_extra[i] != hy_without_extra[i])
                {
                    results_different = 1;
                    break;
                }
            }
            unit_check_scalar(results_different, (int)1);

            // Re-enable extras and verify we get the same result as before
            dy.transfer_from(hy_enabled);

            int32_t enable_extra = 1;

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_enable_extra,
                                                           &enable_extra,
                                                           sizeof(enable_extra),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                    spmv_descr,
                                                    h_alpha,
                                                    matA,
                                                    x,
                                                    h_beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_compute,
                                                    buffer_size,
                                                    dbuffer,
                                                    nullptr));

            host_dense_matrix<Y> hy_re_enabled(hy);
            hy_re_enabled.transfer_from(dy);

            // Verify re-enabled results match original enabled results
            for(size_t i = 0; i < hy_size; ++i)
            {
                unit_check_scalar(hy_with_extra[i], hy_re_enabled[i]);
            }

            // Test error cases: enable/disable when no extras are set
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_clear_extra(handle, spmv_descr, nullptr));

            // These should return errors since no extras are set
            rocsparse_status status;
            int32_t          enable_extra_test = 1;

            status = rocsparse_spmv_set_input(handle,
                                              spmv_descr,
                                              rocsparse_spmv_input_enable_extra,
                                              &enable_extra_test,
                                              sizeof(enable_extra_test),
                                              nullptr);
            unit_check_enum(status, rocsparse_status_invalid_value);

            int32_t disable_extra_test = 0;

            status = rocsparse_spmv_set_input(handle,
                                              spmv_descr,
                                              rocsparse_spmv_input_enable_extra,
                                              &disable_extra_test,
                                              sizeof(disable_extra_test),
                                              nullptr);
            unit_check_enum(status, rocsparse_status_invalid_value);

            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(gamma_vec));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmv_descr(spmv_descr));
            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        }
    }

    static void testing_v2_spmv_multiple_extra(const Arguments& arg)
    {
        J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_datatype     ttype       = get_datatype<T>();

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        // Number of extra vectors to test
        static constexpr int64_t num_extra = 3;
        T gamma_vals[num_extra] = {static_cast<T>(0.5), static_cast<T>(0.3), static_cast<T>(0.7)};

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());
        host_scalar<T> h_gamma[num_extra] = {host_scalar<T>(gamma_vals[0]),
                                             host_scalar<T>(gamma_vals[1]),
                                             host_scalar<T>(gamma_vals[2])};

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);
        device_scalar<T> d_gamma[num_extra] = {device_scalar<T>(h_gamma[0]),
                                               device_scalar<T>(h_gamma[1]),
                                               device_scalar<T>(h_gamma[2])};

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<A> hA;
        {
            int dev;
            CHECK_HIP_ERROR(hipGetDevice(&dev));

            hipDeviceProp_t prop;
            CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

            const bool has_datafile = rocsparse_arguments_has_datafile(arg);
            bool       to_int       = false;
            to_int |= (prop.warpSize == 32);
            to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);
            to_int |= (trans != rocsparse_operation_none && has_datafile);
            to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<A, I, J> matrix_factory(
                arg, arg.unit_check ? to_int : false, full_rank);
            traits::sparse_initialization(matrix_factory, hA, M, N, base);
        }

        if((matrix_type == rocsparse_matrix_type_symmetric && M != N)
           || (matrix_type == rocsparse_matrix_type_triangular && M != N))
        {
            return;
        }

        if(M == 0 || N == 0)
        {
            return;
        }

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx((trans == rocsparse_operation_none) ? N : M, 1);
        rocsparse_matrix_utils::init_exact(hx);
        device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy((trans == rocsparse_operation_none) ? M : N, 1);
        rocsparse_matrix_utils::init_exact(hy);
        device_dense_matrix<Y> dy(hy);

        // Create multiple extra vectors
        host_dense_matrix<X> h_extra[num_extra] = {host_dense_matrix<X>(hy.m, 1),
                                                   host_dense_matrix<X>(hy.m, 1),
                                                   host_dense_matrix<X>(hy.m, 1)};
        for(int64_t i = 0; i < num_extra; ++i)
        {
            rocsparse_matrix_utils::init_exact(h_extra[i]);
        }

        device_dense_matrix<X> d_extra[num_extra] = {device_dense_matrix<X>(h_extra[0]),
                                                     device_dense_matrix<X>(h_extra[1]),
                                                     device_dense_matrix<X>(h_extra[2])};

        // Create the dnvec descriptors directly without assignment to avoid destructor issues
        rocsparse_local_dnvec extra0(d_extra[0]);
        rocsparse_local_dnvec extra1(d_extra[1]);
        rocsparse_local_dnvec extra2(d_extra[2]);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);

        // Test both pointer modes for alpha/beta/gamma
        for(int pointer_mode = 0; pointer_mode < 1; ++pointer_mode)
        {
            if(pointer_mode == 0)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            }
            else
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            }

            // Run buffer size
            rocsparse_spmv_descr spmv_descr;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(
                handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_operation,
                                                           &trans,
                                                           sizeof(trans),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_compute_datatype,
                                                           &ttype,
                                                           sizeof(ttype),
                                                           nullptr));

            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_input(handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_scalar_datatype,
                                                           &ttype,
                                                           sizeof(ttype),
                                                           nullptr));

            // Create gamma dnvec with multiple gamma values
            rocsparse_dnvec_descr gamma_vec;
            CHECK_ROCSPARSE_ERROR(
                rocsparse_create_dnvec_descr(&gamma_vec, num_extra, gamma_vals, ttype));

            rocsparse_const_dnvec_descr z_vecs[num_extra];
            // Get the descriptor directly from each rocsparse_local_dnvec
            rocsparse_local_dnvec* extra_ptrs[num_extra] = {&extra0, &extra1, &extra2};
            for(int64_t i = 0; i < num_extra; ++i)
            {
                rocsparse_dnvec_descr& desc_ref = *extra_ptrs[i];
                z_vecs[i]
                    = desc_ref; // Implicit conversion from rocsparse_dnvec_descr to rocsparse_const_dnvec_descr
            }
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_set_extra(
                handle, spmv_descr, num_extra, gamma_vec, z_vecs, nullptr));

            size_t buffer_size = 0;
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                                spmv_descr,
                                                                matA,
                                                                x,
                                                                y,
                                                                rocsparse_v2_spmv_stage_analysis,
                                                                &buffer_size,
                                                                nullptr));

            void* dbuffer = nullptr;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

            // Run analysis
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv(handle,
                                                    spmv_descr,
                                                    pointer_mode == 0 ? h_alpha : d_alpha,
                                                    matA,
                                                    x,
                                                    pointer_mode == 0 ? h_beta : d_beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_analysis,
                                                    buffer_size,
                                                    dbuffer,
                                                    nullptr));

            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
            dbuffer = nullptr;
            CHECK_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size(handle,
                                                                spmv_descr,
                                                                matA,
                                                                x,
                                                                y,
                                                                rocsparse_v2_spmv_stage_compute,
                                                                &buffer_size,
                                                                nullptr));
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

            if(arg.unit_check)
            {
                // Run solve
                CHECK_ROCSPARSE_ERROR(
                    testing::rocsparse_v2_spmv(handle,
                                               spmv_descr,
                                               pointer_mode == 0 ? h_alpha : d_alpha,
                                               matA,
                                               x,
                                               pointer_mode == 0 ? h_beta : d_beta,
                                               y,
                                               rocsparse_v2_spmv_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr));

                host_dense_matrix<Y> hy_copy(hy);
                traits::host_calculation(trans,
                                         pointer_mode == 0 ? h_alpha : d_alpha,
                                         hA,
                                         hx,
                                         pointer_mode == 0 ? h_beta : d_beta,
                                         hy,
                                         alg,
                                         matrix_type);

                const size_t hy_size = (trans == rocsparse_operation_none) ? M : N;

                // Add contribution from all extra vectors: y += gamma[i] * z[i]
                for(int64_t i = 0; i < num_extra; ++i)
                {
                    for(size_t j = 0; j < hy_size; ++j)
                    {
                        hy[j] += gamma_vals[i] * h_extra[i][j];
                    }
                }

                hy.near_check(dy);

                if(ROCSPARSE_REPRODUCIBILITY)
                {
                    rocsparse_reproducibility::save(pointer_mode == 0
                                                        ? "Y_pointer_mode_host_multiple_extra"
                                                        : "Y_pointer_mode_device_multiple_extra",
                                                    dy);
                }

                dy.transfer_from(hy_copy);
            }

            // Clear extra vectors after computation
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv_clear_extra(handle, spmv_descr, nullptr));

            if(arg.timing)
            {
                const double gpu_time_used
                    = rocsparse_clients::run_benchmark(arg,
                                                       rocsparse_v2_spmv,
                                                       handle,
                                                       spmv_descr,
                                                       pointer_mode == 0 ? h_alpha : d_alpha,
                                                       matA,
                                                       x,
                                                       pointer_mode == 0 ? h_beta : d_beta,
                                                       y,
                                                       rocsparse_v2_spmv_stage_compute,
                                                       buffer_size,
                                                       dbuffer,
                                                       nullptr);

                const double gflop_count = traits::gflop_count(hA, *h_beta != static_cast<T>(0));
                const double gbyte_count = traits::byte_count(hA, *h_beta != static_cast<T>(0));

                const double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
                const double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

                if(arg.sparsity_pattern_statistics)
                {
                    int64_t min_nnz_row;
                    int64_t median_nnz_row;
                    int64_t max_nnz_row;
                    rocsparse_matrix_statistics::get_nnz_per_row(
                        dA, min_nnz_row, median_nnz_row, max_nnz_row);

                    int64_t min_nnz_col;
                    int64_t median_nnz_col;
                    int64_t max_nnz_col;
                    rocsparse_matrix_statistics::get_nnz_per_column(
                        dA, min_nnz_col, median_nnz_col, max_nnz_col);
                    traits::display_info(arg,
                                         display_key_t::trans_A,
                                         rocsparse_operation2string(trans),
                                         dA,
                                         display_key_t::min_nnz_per_row,
                                         min_nnz_row,
                                         display_key_t::max_nnz_per_row,
                                         max_nnz_row,
                                         display_key_t::median_nnz_per_row,
                                         median_nnz_row,
                                         display_key_t::min_nnz_per_col,
                                         min_nnz_col,
                                         display_key_t::max_nnz_per_col,
                                         max_nnz_col,
                                         display_key_t::median_nnz_per_col,
                                         median_nnz_col,
                                         display_key_t::alpha,
                                         *h_alpha,
                                         display_key_t::beta,
                                         *h_beta,
                                         display_key_t::algorithm,
                                         rocsparse_spmvalg2string(alg),
                                         display_key_t::gflops,
                                         gpu_gflops,
                                         display_key_t::bandwidth,
                                         gpu_gbyte,
                                         display_key_t::time_ms,
                                         get_gpu_time_msec(gpu_time_used));
                }
                else
                {
                    traits::display_info(arg,
                                         display_key_t::trans_A,
                                         rocsparse_operation2string(trans),
                                         dA,
                                         display_key_t::alpha,
                                         *h_alpha,
                                         display_key_t::beta,
                                         *h_beta,
                                         display_key_t::algorithm,
                                         rocsparse_spmvalg2string(alg),
                                         display_key_t::gflops,
                                         gpu_gflops,
                                         display_key_t::bandwidth,
                                         gpu_gbyte,
                                         display_key_t::time_ms,
                                         get_gpu_time_msec(gpu_time_used));
                }
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(gamma_vec));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmv_descr(spmv_descr));
            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        }
    }
};

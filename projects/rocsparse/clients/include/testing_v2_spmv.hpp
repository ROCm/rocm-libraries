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
};

/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_csrsv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    const T h_alpha = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle            = local_handle;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             m                 = safe_size;
    rocsparse_int             nnz               = safe_size;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  csr_val           = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_mat_info        info              = local_info;
    const T*                  x                 = (const T*)0x4;
    T*                        y                 = (T*)0x4;
    rocsparse_analysis_policy analysis          = rocsparse_analysis_policy_reuse;
    rocsparse_solve_policy    solve             = rocsparse_solve_policy_auto;
    rocsparse_solve_policy    policy            = rocsparse_solve_policy_auto;
    size_t*                   buffer_size       = (size_t*)0x4;
    void*                     temp_buffer       = (void*)0x4;

#define PARAMS_BUFFER_SIZE \
    handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size

#define PARAMS_ANALYSIS                                                                     \
    handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, analysis, solve, \
        temp_buffer

#define PARAMS_SOLVE                                                                             \
    handle, trans, m, nnz, alpha_device_host, descr, csr_val, csr_row_ptr, csr_col_ind, info, x, \
        y, policy, temp_buffer

    bad_arg_analysis(rocsparse_csrsv_buffer_size<T>, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_csrsv_analysis<T>, PARAMS_ANALYSIS);
    bad_arg_analysis(rocsparse_csrsv_solve<T>, PARAMS_SOLVE);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general
           && matrix_type != rocsparse_matrix_type_triangular)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(PARAMS_SOLVE),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(PARAMS_SOLVE),
                            rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE

    // Test rocsparse_csrsv_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(nullptr, descr, info, &position),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, nullptr, &position),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(nullptr, descr, info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, nullptr, info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, descr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csrsv(const Arguments& arg)
{
    auto tol = get_near_check_tol<T>(arg);

    rocsparse_int             M     = arg.M;
    rocsparse_int             N     = arg.N;
    rocsparse_operation       trans = arg.transA;
    rocsparse_diag_type       diag  = arg.diag;
    rocsparse_fill_mode       uplo  = arg.uplo;
    rocsparse_analysis_policy apol  = arg.apol;
    rocsparse_solve_policy    spol  = arg.spol;
    rocsparse_index_base      base  = arg.baseA;

    host_scalar<T> h_alpha(arg.get_alpha<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    hipStream_t stream{};
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix diag type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));

    // Set matrix fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

#define PARAMS_BUFFER_SIZE(A_) \
    handle, trans, A_.m, A_.nnz, descr, A_.val, A_.ptr, A_.ind, info, &buffer_size
#define PARAMS_ANALYSIS(A_) \
    handle, trans, A_.m, A_.nnz, descr, A_.val, A_.ptr, A_.ind, info, apol, spol, dbuffer
#define PARAMS_SOLVE(alpha_, A_, x_, y_) \
    handle, trans, A_.m, A_.nnz, alpha_, descr, A_.val, A_.ptr, A_.ind, info, x_, y_, spol, dbuffer

    // Argument sanity check before allocating invalid memory
    if(M <= 0)
    {
        size_t        buffer_size;
        rocsparse_int pivot;

        device_vector<T>     dx, dy, dbuffer;
        device_csr_matrix<T> dA;
        dA.m   = M;
        dA.n   = M;
        dA.nnz = 10;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, &pivot),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, descr, info),
                                rocsparse_status_success);

        return;
    }

    // Sample matrix
    host_csr_matrix<T> hA;

    {
        static constexpr bool       to_int    = false;
        static constexpr bool       full_rank = true;
        rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);
        matrix_factory.init_csr(hA, M, N);
    }

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    host_dense_matrix<T> hx(M, 1);
    rocsparse_matrix_utils::init(hx);

    device_csr_matrix<T>       dA(hA);
    device_dense_matrix<T>     dx(hx), dy(M, 1);
    host_scalar<rocsparse_int> h_analysis_pivot, h_solve_pivot;

    // Obtain required buffer size
    void* dbuffer;
    {
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    }

    if(arg.unit_check)
    {
        host_scalar<rocsparse_int> analysis_no_pivot(-1);
        host_dense_matrix<T>       hy(M, 1);
        // CPU csrsv
        host_csrsv<rocsparse_int, rocsparse_int, T>(trans,
                                                    hA.m,
                                                    hA.nnz,
                                                    *h_alpha,
                                                    hA.ptr,
                                                    hA.ind,
                                                    hA.val,
                                                    hx,
                                                    (int64_t)1,
                                                    hy,
                                                    diag,
                                                    uplo,
                                                    base,
                                                    h_analysis_pivot,
                                                    h_solve_pivot);

        // Pointer mode host
        {
            host_scalar<rocsparse_int> analysis_pivot;
            host_scalar<rocsparse_int> solve_pivot;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            //
            // CHECK IF DEFAULT ZERO PIVOT IS -1
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, analysis_pivot),
                                    rocsparse_status_success);
            analysis_no_pivot.unit_check(analysis_pivot);

            //
            // Call before analysis
            //
            EXPECT_ROCSPARSE_STATUS(
                testing::rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                rocsparse_status_invalid_pointer);

            // Call it twice.
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            {
                auto st = rocsparse_csrsv_zero_pivot(handle, descr, info, analysis_pivot);
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                EXPECT_ROCSPARSE_STATUS(st,
                                        (*analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                                : rocsparse_status_success);
            }

            CHECK_ROCSPARSE_ERROR(
                testing::rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
            {
                auto st = rocsparse_csrsv_zero_pivot(handle, descr, info, solve_pivot);
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                EXPECT_ROCSPARSE_STATUS(st,
                                        (*solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);
            }
            h_analysis_pivot.unit_check(analysis_pivot);
            h_solve_pivot.unit_check(solve_pivot);
        }

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            hy.near_check(dy, tol);
        }

        //
        // RESET MAT INFO.
        //
        info.reset();

        {
            device_scalar<rocsparse_int> d_analysis_pivot;
            device_scalar<rocsparse_int> d_solve_pivot;
            device_scalar<T>             d_alpha(h_alpha);

            // Pointer mode device
            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            //
            // CHECK IF DEFAULT ZERO PIVOT IS -1
            //
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_csrsv_zero_pivot(handle, descr, info, d_analysis_pivot),
                rocsparse_status_success);
            analysis_no_pivot.unit_check(d_analysis_pivot);

            //
            // Call before analysis
            //
            EXPECT_ROCSPARSE_STATUS(
                testing::rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                rocsparse_status_invalid_pointer);

            // Call it twice.
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_csrsv_zero_pivot(handle, descr, info, d_analysis_pivot),
                (*h_analysis_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success);
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            CHECK_ROCSPARSE_ERROR(
                testing::rocsparse_csrsv_solve<T>(PARAMS_SOLVE(d_alpha, dA, dx, dy)));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, d_solve_pivot),
                                    (*h_solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            h_analysis_pivot.unit_check(d_analysis_pivot);
            h_solve_pivot.unit_check(d_solve_pivot);
        }

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            hy.near_check(dy, tol);
        }

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        info.reset();

        {
            void*  buffer = nullptr;
            size_t buffer_size;
            int    boost       = arg.numericboost;
            T      h_boost_tol = static_cast<T>(arg.boosttol);
            T      h_boost_val = arg.get_boostval<T>();

            rocsparse_matrix_utils::csrilu0<T>(descr,
                                               dA,
                                               info,
                                               apol,
                                               spol,
                                               boost,
                                               h_boost_val,
                                               h_boost_tol,
                                               &buffer_size,
                                               buffer,
                                               rocsparse_matrix_utils::csrilu0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            rocsparse_matrix_utils::csrilu0<T>(descr,
                                               dA,
                                               info,
                                               apol,
                                               spol,
                                               boost,
                                               h_boost_val,
                                               h_boost_tol,
                                               &buffer_size,
                                               buffer,
                                               rocsparse_matrix_utils::csrilu0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        info.reset();

        {
            void*  buffer = nullptr;
            size_t buffer_size;
            rocsparse_matrix_utils::csric0<T>(descr,
                                              dA,
                                              info,
                                              apol,
                                              spol,
                                              &buffer_size,
                                              buffer,
                                              rocsparse_matrix_utils::csric0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            rocsparse_matrix_utils::csric0<T>(descr,
                                              dA,
                                              info,
                                              apol,
                                              spol,
                                              &buffer_size,
                                              buffer,
                                              rocsparse_matrix_utils::csric0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        info.reset();

        {
            void*  buffer = nullptr;
            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_buffer_size<T>(handle,
                                                                 rocsparse_operation_transpose,
                                                                 rocsparse_operation_none,
                                                                 dA.m,
                                                                 1,
                                                                 dA.nnz,
                                                                 h_alpha,
                                                                 descr,
                                                                 dA.val,
                                                                 dA.ptr,
                                                                 dA.ind,
                                                                 dx,
                                                                 dA.m,
                                                                 info,
                                                                 spol,
                                                                 &buffer_size));
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_analysis<T>(handle,
                                                              rocsparse_operation_transpose,
                                                              rocsparse_operation_none,
                                                              dA.m,
                                                              1,
                                                              dA.nnz,
                                                              h_alpha,
                                                              descr,
                                                              dA.val,
                                                              dA.ptr,
                                                              dA.ind,
                                                              dx,
                                                              dA.m,
                                                              info,
                                                              apol,
                                                              spol,
                                                              buffer));

            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        info.reset();

        {
            void*  buffer = nullptr;
            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_buffer_size<T>(handle,
                                                                 rocsparse_operation_none,
                                                                 rocsparse_operation_none,
                                                                 dA.m,
                                                                 1,
                                                                 dA.nnz,
                                                                 h_alpha,
                                                                 descr,
                                                                 dA.val,
                                                                 dA.ptr,
                                                                 dA.ind,
                                                                 dx,
                                                                 dA.m,
                                                                 info,
                                                                 spol,
                                                                 &buffer_size));
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_analysis<T>(handle,
                                                              rocsparse_operation_none,
                                                              rocsparse_operation_none,
                                                              dA.m,
                                                              1,
                                                              dA.nnz,
                                                              h_alpha,
                                                              descr,
                                                              dA.val,
                                                              dA.ptr,
                                                              dA.ind,
                                                              dx,
                                                              dA.m,
                                                              info,
                                                              apol,
                                                              spol,
                                                              buffer));

            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csrsv_zero_pivot(handle, descr, info, h_analysis_pivot));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_zero_pivot(handle, descr, info, h_solve_pivot));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descr, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gflop_count = csrsv_gflop_count(M, dA.nnz, diag);
        double gbyte_count = csrsv_gbyte_count<T>(M, dA.nnz);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::pivot,
                            std::min(*h_analysis_pivot, *h_solve_pivot),
                            display_key_t::trans,
                            rocsparse_operation2string(trans),
                            display_key_t::diag_type,
                            rocsparse_diagtype2string(diag),
                            display_key_t::fill_mode,
                            rocsparse_fillmode2string(uplo),
                            display_key_t::analysis_policy,
                            rocsparse_analysis2string(apol),
                            display_key_t::solve_policy,
                            rocsparse_solve2string(spol),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::analysis_time_ms,
                            get_gpu_time_msec(gpu_analysis_time_used),
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Clear csrsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descr, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrsv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrsv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrsv_extra(const Arguments& arg) {}

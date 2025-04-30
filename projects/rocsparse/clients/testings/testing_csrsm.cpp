/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_csrsm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle      = local_handle;
    rocsparse_operation       trans_A     = rocsparse_operation_none;
    rocsparse_operation       trans_B     = rocsparse_operation_none;
    rocsparse_int             m           = safe_size;
    rocsparse_int             nrhs        = safe_size;
    rocsparse_int             nnz         = safe_size;
    const T*                  alpha       = &h_alpha;
    const rocsparse_mat_descr descr       = local_descr;
    const T*                  csr_val     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    T*                        B           = (T*)0x4;
    rocsparse_int             ldb         = safe_size;
    rocsparse_mat_info        info        = local_info;
    rocsparse_analysis_policy analysis    = rocsparse_analysis_policy_force;
    rocsparse_solve_policy    solve       = rocsparse_solve_policy_auto;
    rocsparse_solve_policy    policy      = rocsparse_solve_policy_auto;
    size_t                    local_buffer_size;
    size_t*                   buffer_size = &local_buffer_size;
    void*                     temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                      \
    handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, \
        ldb, info, policy, buffer_size
    bad_arg_analysis(rocsparse_csrsm_buffer_size<T>, PARAMS_BUFFER_SIZE);

#define PARAMS_ANALYSIS                                                                         \
    handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, \
        ldb, info, analysis, solve, temp_buffer
    bad_arg_analysis(rocsparse_csrsm_analysis<T>, PARAMS_ANALYSIS);

#define PARAMS_SOLVE                                                                            \
    handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, \
        ldb, info, policy, temp_buffer
    bad_arg_analysis(rocsparse_csrsm_solve<T>, PARAMS_SOLVE);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_solve<T>(PARAMS_SOLVE),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_solve<T>(PARAMS_SOLVE),
                            rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // Test rocsparse_csrsm_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_zero_pivot(nullptr, info, &position),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_zero_pivot(handle, nullptr, &position),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_zero_pivot(handle, info, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsm_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE
}

template <typename T>
void testing_csrsm(const Arguments& arg)
{
    rocsparse_operation       transA    = arg.transA;
    rocsparse_operation       transB    = arg.transB;
    rocsparse_int             M         = arg.M;
    rocsparse_int             nrhs      = arg.K;
    rocsparse_diag_type       diag      = arg.diag;
    rocsparse_fill_mode       uplo      = arg.uplo;
    rocsparse_analysis_policy apol      = arg.apol;
    rocsparse_solve_policy    spol      = arg.spol;
    rocsparse_index_base      base      = arg.baseA;
    static constexpr bool     full_rank = true;

    rocsparse_matrix_factory<T> matrix_factory(arg, false, full_rank);

    host_scalar<T> h_alpha(arg.get_alpha<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Create matrix info

    // Set matrix diag type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));

    // Set matrix fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    host_csr_matrix<T> hcsr;
    matrix_factory.init_csr(hcsr);

    //
    // Scale the matrix.
    //
    hcsr.scale();

    rocsparse_int nnz = hcsr.nnz;
    M                 = hcsr.m;

    rocsparse_int hB_m = (transB == rocsparse_operation_none) ? M : nrhs;
    rocsparse_int hB_n = (transB == rocsparse_operation_none) ? nrhs : M;

    host_dense_matrix<T> hB(hB_m, hB_n);
    rocsparse_init<T>(hB, 1, M * nrhs, 1);

    host_scalar<rocsparse_int> h_analysis_pivot;
    host_scalar<rocsparse_int> h_solve_pivot;

    // Allocate device memory
    device_csr_matrix<T>   dcsr(hcsr);
    device_dense_matrix<T> dB(hB);

#define CALL_BUFFER_SIZE(alpha)                                    \
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_buffer_size<T>(handle,   \
                                                         transA,   \
                                                         transB,   \
                                                         dcsr.m,   \
                                                         nrhs,     \
                                                         dcsr.nnz, \
                                                         alpha,    \
                                                         descr,    \
                                                         dcsr.val, \
                                                         dcsr.ptr, \
                                                         dcsr.ind, \
                                                         dB,       \
                                                         dB.ld,    \
                                                         info,     \
                                                         spol,     \
                                                         &buffer_size))

#define CALL_ANALYSIS(alpha)                                    \
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_analysis<T>(handle,   \
                                                      transA,   \
                                                      transB,   \
                                                      dcsr.m,   \
                                                      nrhs,     \
                                                      dcsr.nnz, \
                                                      alpha,    \
                                                      descr,    \
                                                      dcsr.val, \
                                                      dcsr.ptr, \
                                                      dcsr.ind, \
                                                      dB,       \
                                                      dB.ld,    \
                                                      info,     \
                                                      apol,     \
                                                      spol,     \
                                                      dbuffer))

#define CALL_SOLVE(alpha)                                    \
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_solve<T>(handle,   \
                                                   transA,   \
                                                   transB,   \
                                                   M,        \
                                                   nrhs,     \
                                                   nnz,      \
                                                   alpha,    \
                                                   descr,    \
                                                   dcsr.val, \
                                                   dcsr.ptr, \
                                                   dcsr.ind, \
                                                   dB,       \
                                                   dB.ld,    \
                                                   info,     \
                                                   spol,     \
                                                   dbuffer))

#define CALL_TESTING_SOLVE(alpha)                                     \
    CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrsm_solve<T>(handle,   \
                                                            transA,   \
                                                            transB,   \
                                                            M,        \
                                                            nrhs,     \
                                                            nnz,      \
                                                            alpha,    \
                                                            descr,    \
                                                            dcsr.val, \
                                                            dcsr.ptr, \
                                                            dcsr.ind, \
                                                            dB,       \
                                                            dB.ld,    \
                                                            info,     \
                                                            spol,     \
                                                            dbuffer))

    // Obtain required buffer size

    size_t buffer_size;
    CALL_BUFFER_SIZE(h_alpha);

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    if(arg.unit_check)
    {
        host_scalar<rocsparse_int> analysis_pivot;
        host_scalar<rocsparse_int> solve_pivot;
        rocsparse_status           status;

        //
        // CALL ANALYSIS WITH HOST MODE
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CALL_ANALYSIS(h_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        //
        // GET ANALYSIS PIVOT INFORMATION
        //
        status = rocsparse_csrsm_zero_pivot(handle, info, analysis_pivot);
        EXPECT_ROCSPARSE_STATUS(status,
                                (*analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                        : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        //
        // A SECOND TIME FOR FOR CODE COVERAGE.
        //
        CALL_ANALYSIS(h_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        //
        // CALL SOLVE WITH HOST MODE
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CALL_TESTING_SOLVE(h_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        //
        // GET SOLVE PIVOT INFORMATION
        //
        status = rocsparse_csrsm_zero_pivot(handle, info, solve_pivot);
        EXPECT_ROCSPARSE_STATUS(
            status, (*solve_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        {
            //
            // CALL HOST CALCULATION
            //
            host_dense_matrix<T> hB_copy(hB);
            host_csrsm<rocsparse_int, rocsparse_int, T>(M,
                                                        nrhs,
                                                        nnz,
                                                        transA,
                                                        transB,
                                                        *h_alpha,
                                                        hcsr.ptr,
                                                        hcsr.ind,
                                                        hcsr.val,
                                                        hB,
                                                        hB.ld,
                                                        rocsparse_order_column,
                                                        diag,
                                                        uplo,
                                                        base,
                                                        h_analysis_pivot,
                                                        h_solve_pivot);

            //
            // CHECK PIVOTS
            //
            h_analysis_pivot.unit_check(analysis_pivot);
            h_solve_pivot.unit_check(solve_pivot);

            //
            // CHECK SOLUTION VECTOR IF NO PIVOT HAS BEEN FOUND
            //
            if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
            {
                hB.near_check(dB);
            }

            dB = hB_copy;
        }

        //
        // COPY ALPHA.
        //
        device_dense_matrix<T> d_alpha(h_alpha);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        //
        // RESET MAT INFO.
        //
        info.reset();

        //
        // CALL BUFFER SIZE WITH DEVICE MODE.
        //
        size_t buffer_size_bis = buffer_size;
        CALL_BUFFER_SIZE(d_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        unit_check_scalar<size_t>(buffer_size, buffer_size_bis);

        //
        // CALL ANALYSIS WITH DEVICE MODE
        //
        CALL_ANALYSIS(d_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        //
        // CHECK PIVOT ANALYSIS
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        device_scalar<rocsparse_int> d_analysis_pivot;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_zero_pivot(handle, info, d_analysis_pivot),
                                (*h_analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        h_analysis_pivot.unit_check(d_analysis_pivot);

        //
        // CALL SOLVE WITH DEVICE MODE
        //
        CALL_TESTING_SOLVE(d_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        //
        // CHECK PIVOT SOLVE
        //
        device_scalar<rocsparse_int> d_solve_pivot;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsm_zero_pivot(handle, info, d_solve_pivot),
                                (*h_solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                       : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        h_solve_pivot.unit_check(d_solve_pivot);

        //
        // CHECK SOLUTION VECTOR IF NO PIVOT HAS BEEN FOUND
        //
        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            hB.near_check(dB);
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
            CALL_ANALYSIS(h_alpha);
            CALL_SOLVE(h_alpha);
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_clear(handle, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CALL_ANALYSIS(h_alpha);
        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CALL_SOLVE(h_alpha);
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gflop_count = csrsv_gflop_count(M, nnz, diag) * nrhs;
        double gbyte_count = csrsv_gbyte_count<T>(M, nnz) * nrhs;

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::nrhs,
                            nrhs,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::pivot,
                            std::min(*h_analysis_pivot, *h_solve_pivot),
                            display_key_t::trans_A,
                            rocsparse_operation2string(transA),
                            display_key_t::trans_B,
                            rocsparse_operation2string(transB),
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

    // Clear csrsm meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsm_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrsm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrsm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrsm_extra(const Arguments& arg) {}

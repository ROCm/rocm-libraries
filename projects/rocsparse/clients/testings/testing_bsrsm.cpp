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
#include "testing.hpp"

#include "rocsparse_enum.hpp"

template <typename T>
void testing_bsrsm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    T             h_alpha = static_cast<T>(1);
    size_t        h_buffer_size;
    rocsparse_int h_position;

    // Local declaration
    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_operation       trans_A     = rocsparse_operation_none;
    rocsparse_operation       trans_X     = rocsparse_operation_none;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nrhs        = safe_size;
    rocsparse_int             nnzb        = safe_size;
    const T*                  alpha       = &h_alpha;
    rocsparse_mat_descr       descr       = local_descr;
    const T*                  bsr_val     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const T*                  B           = (const T*)0x4;
    rocsparse_int             ldb         = mb * block_dim;
    T*                        X           = (T*)0x4;
    rocsparse_int             ldx         = mb * block_dim;
    rocsparse_mat_info        info        = local_info;
    rocsparse_analysis_policy analysis    = rocsparse_analysis_policy_force;
    rocsparse_solve_policy    solve       = rocsparse_solve_policy_auto;
    rocsparse_solve_policy    policy      = rocsparse_solve_policy_auto;
    size_t*                   buffer_size = &h_buffer_size;
    void*                     temp_buffer = (void*)0x4;
    rocsparse_int*            position    = &h_position;

    // bsrsm_buffer_size
#define PARAMS_BUFFER_SIZE                                                                   \
    handle, dir, trans_A, trans_X, mb, nrhs, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, \
        block_dim, info, buffer_size
    bad_arg_analysis(rocsparse_bsrsm_buffer_size<T>, PARAMS_BUFFER_SIZE);

    // Invalid size
    {
        auto tmp  = block_dim;
        block_dim = 0;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                rocsparse_status_invalid_size);
        block_dim = tmp;
    }

    // bsrsm_analysis
#define PARAMS_ANALYSIS                                                                      \
    handle, dir, trans_A, trans_X, mb, nrhs, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, \
        block_dim, info, analysis, solve, temp_buffer
    bad_arg_analysis(rocsparse_bsrsm_analysis<T>, PARAMS_ANALYSIS);

    // Invalid size
    {
        auto tmp  = block_dim;
        block_dim = 0;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_analysis<T>(PARAMS_ANALYSIS),
                                rocsparse_status_invalid_size);
        block_dim = tmp;
    }

    // bsrsm_solve
#define PARAMS_SOLVE                                                                   \
    handle, dir, trans_A, trans_X, mb, nrhs, nnzb, alpha, descr, bsr_val, bsr_row_ptr, \
        bsr_col_ind, block_dim, info, B, ldb, X, ldx, policy, temp_buffer
    bad_arg_analysis(rocsparse_bsrsm_solve<T>, PARAMS_SOLVE);

    // Invalid size
    {
        auto tmp  = block_dim;
        block_dim = 0;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                rocsparse_status_invalid_size);
        block_dim = tmp;
    }
    {
        auto tmp = ldb;
        ldb      = safe_size / 2;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                rocsparse_status_invalid_size);
        ldb = tmp;
    }
    {
        auto tmp = ldx;
        ldx      = safe_size / 2;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                rocsparse_status_invalid_size);
        ldx = tmp;
    }
    {
        auto tmp = ldb;
        ldb      = safe_size / 2;
        trans_X  = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                rocsparse_status_invalid_size);
        ldb = tmp;
        // trans_X = rocsparse_operation_none;
    }
    {
        auto tmp = ldx;
        ldx      = safe_size / 2;
        trans_X  = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                rocsparse_status_invalid_size);
        ldx     = tmp;
        trans_X = rocsparse_operation_none;
    }

    // bsrsm_zero_pivot
    bad_arg_analysis(rocsparse_bsrsm_zero_pivot, handle, info, position);

    // bsrsm_clear
    bad_arg_analysis(rocsparse_bsrsm_clear, handle, info);

    // Matrix types different from general
    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(PARAMS_SOLVE),
                            rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_buffer_size<T>(handle,
                                                           dir,
                                                           trans_A,
                                                           trans_X,
                                                           mb,
                                                           nrhs,
                                                           nnzb,
                                                           descr,
                                                           nullptr,
                                                           bsr_row_ptr,
                                                           nullptr,
                                                           block_dim,
                                                           info,
                                                           buffer_size),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_analysis<T>(handle,
                                                        dir,
                                                        trans_A,
                                                        trans_X,
                                                        mb,
                                                        nrhs,
                                                        nnzb,
                                                        descr,
                                                        nullptr,
                                                        bsr_row_ptr,
                                                        nullptr,
                                                        block_dim,
                                                        info,
                                                        analysis,
                                                        solve,
                                                        temp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_solve<T>(handle,
                                                     dir,
                                                     trans_A,
                                                     trans_X,
                                                     mb,
                                                     nrhs,
                                                     nnzb,
                                                     alpha,
                                                     descr,
                                                     nullptr,
                                                     bsr_row_ptr,
                                                     nullptr,
                                                     block_dim,
                                                     info,
                                                     B,
                                                     ldb,
                                                     X,
                                                     ldx,
                                                     solve,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrsm(const Arguments& arg)
{
    auto                      tol       = get_near_check_tol<T>(arg);
    rocsparse_int             m         = arg.M;
    rocsparse_int             nrhs      = arg.K;
    rocsparse_int             block_dim = arg.block_dim;
    rocsparse_operation       trans_A   = arg.transA;
    rocsparse_operation       trans_X   = arg.transB;
    rocsparse_direction       dir       = arg.direction;
    rocsparse_diag_type       diag      = arg.diag;
    rocsparse_fill_mode       uplo      = arg.uplo;
    rocsparse_analysis_policy apol      = arg.apol;
    rocsparse_solve_policy    spol      = arg.spol;
    rocsparse_index_base      base      = arg.baseA;

    // BSR dimension
    rocsparse_int mb = (m + block_dim - 1) / block_dim;

    // Scalar
    host_scalar<T> h_alpha(arg.get_alpha<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

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

    // Allocate host memory for BSR matrix A
    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    matrix_factory.init_bsr(hA, dA, mb, mb, base);

    m = mb * dA.row_block_dim;

    // RHS matrix B
    host_dense_matrix<T> hB((trans_X == rocsparse_operation_none) ? m : nrhs,
                            (trans_X == rocsparse_operation_none) ? nrhs : m);
    rocsparse_matrix_utils::init(hB, arg.convert_to_int);
    device_dense_matrix<T> dB(hB);

    // Solution matrix X
    host_dense_matrix<T> hX_gold((trans_X == rocsparse_operation_none) ? m : nrhs,
                                 (trans_X == rocsparse_operation_none) ? nrhs : m);
    rocsparse_matrix_utils::init(hX_gold, arg.convert_to_int);
    device_dense_matrix<T> dX(hX_gold);

    host_scalar<rocsparse_int> analysis_pivot_gold;
    host_scalar<rocsparse_int> solve_pivot_gold;

#define CALL_BUFFER_SIZE                                            \
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsm_buffer_size<T>(handle,    \
                                                         dir,       \
                                                         trans_A,   \
                                                         trans_X,   \
                                                         mb,        \
                                                         nrhs,      \
                                                         dA.nnzb,   \
                                                         descr,     \
                                                         dA.val,    \
                                                         dA.ptr,    \
                                                         dA.ind,    \
                                                         block_dim, \
                                                         info,      \
                                                         &buffer_size));

#define CALL_ANALYSIS                                            \
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsm_analysis<T>(handle,    \
                                                      dir,       \
                                                      trans_A,   \
                                                      trans_X,   \
                                                      mb,        \
                                                      nrhs,      \
                                                      dA.nnzb,   \
                                                      descr,     \
                                                      dA.val,    \
                                                      dA.ptr,    \
                                                      dA.ind,    \
                                                      block_dim, \
                                                      info,      \
                                                      apol,      \
                                                      spol,      \
                                                      dbuffer))

#define CALL_SOLVE(alpha)                                     \
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsm_solve<T>(handle,    \
                                                   dir,       \
                                                   trans_A,   \
                                                   trans_X,   \
                                                   mb,        \
                                                   nrhs,      \
                                                   dA.nnzb,   \
                                                   alpha,     \
                                                   descr,     \
                                                   dA.val,    \
                                                   dA.ptr,    \
                                                   dA.ind,    \
                                                   block_dim, \
                                                   info,      \
                                                   dB,        \
                                                   dB.ld,     \
                                                   dX,        \
                                                   dX.ld,     \
                                                   spol,      \
                                                   dbuffer))

#define CALL_TESTING_SOLVE(alpha)                                      \
    CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrsm_solve<T>(handle,    \
                                                            dir,       \
                                                            trans_A,   \
                                                            trans_X,   \
                                                            mb,        \
                                                            nrhs,      \
                                                            dA.nnzb,   \
                                                            alpha,     \
                                                            descr,     \
                                                            dA.val,    \
                                                            dA.ptr,    \
                                                            dA.ind,    \
                                                            block_dim, \
                                                            info,      \
                                                            dB,        \
                                                            dB.ld,     \
                                                            dX,        \
                                                            dX.ld,     \
                                                            spol,      \
                                                            dbuffer))

    // Obtain required buffer size
    size_t buffer_size;

    CALL_BUFFER_SIZE;

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // HOST MODE
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        host_scalar<rocsparse_int> analysis_pivot;
        host_scalar<rocsparse_int> solve_pivot;

        // bsrsm_analysis
        CALL_ANALYSIS;
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        // Obtain pivot information
        {
            auto st = rocsparse_bsrsm_zero_pivot(handle, info, analysis_pivot);
            EXPECT_ROCSPARSE_STATUS(
                st,
                ((*analysis_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success));
        }
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // bsrsm_solve
        CALL_TESTING_SOLVE(h_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Obtain pivot information
        {
            auto st = rocsparse_bsrsm_zero_pivot(handle, info, solve_pivot);
            EXPECT_ROCSPARSE_STATUS(
                st, (*solve_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success);
        }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        // host_bsrsm
        host_bsrsm<T>(mb,
                      nrhs,
                      hA.nnzb,
                      dir,
                      trans_A,
                      trans_X,
                      *h_alpha,
                      hA.ptr,
                      hA.ind,
                      hA.val,
                      block_dim,
                      hB,
                      hB.ld,
                      hX_gold,
                      hX_gold.ld,
                      diag,
                      uplo,
                      base,
                      analysis_pivot_gold,
                      solve_pivot_gold);

        // Check pivots
        analysis_pivot_gold.unit_check(analysis_pivot);
        solve_pivot_gold.unit_check(solve_pivot);

        // Check solution matrix if no pivot has been found
        if(*analysis_pivot_gold == -1 && *solve_pivot_gold == -1)
        {
            hX_gold.near_check(dX, tol);
        }

        // DEVICE MODE
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        // Copy alpha to device
        device_dense_matrix<T> d_alpha(h_alpha);

        // Reset mat info
        info.reset();

        size_t buffer_size_gold = buffer_size;
        CALL_BUFFER_SIZE;
        unit_check_scalar(buffer_size, buffer_size_gold);

        // bsrsm_analysis
        CALL_ANALYSIS;
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Obtain pivot information
        device_scalar<rocsparse_int> d_analysis_pivot;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_zero_pivot(handle, info, d_analysis_pivot),
                                (*analysis_pivot_gold != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // bsrsm_solve
        CALL_TESTING_SOLVE(d_alpha);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Obtain pivot information
        device_scalar<rocsparse_int> d_solve_pivot;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsm_zero_pivot(handle, info, d_solve_pivot),
                                (*solve_pivot_gold != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Check pivots
        analysis_pivot_gold.unit_check(d_analysis_pivot);
        solve_pivot_gold.unit_check(d_solve_pivot);

        // Check solution matrix if no pivot has been found
        if(*analysis_pivot_gold == -1 && *solve_pivot_gold == -1)
        {
            hX_gold.near_check(dX, tol);
        }
    }

    if(arg.timing)
    {

        device_dense_matrix<T> d_alpha(h_alpha);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_analysis_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_bsrsm_analysis<T>,
                                               handle,
                                               dir,
                                               trans_A,
                                               trans_X,
                                               mb,
                                               nrhs,
                                               dA.nnzb,
                                               descr,
                                               dA.val,
                                               dA.ptr,
                                               dA.ind,
                                               block_dim,
                                               info,
                                               apol,
                                               spol,
                                               dbuffer);

        rocsparse_bsrsm_zero_pivot(handle, info, analysis_pivot_gold);

        const double gpu_solve_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_bsrsm_solve<T>,
                                               handle,
                                               dir,
                                               trans_A,
                                               trans_X,
                                               mb,
                                               nrhs,
                                               dA.nnzb,
                                               d_alpha,
                                               descr,
                                               dA.val,
                                               dA.ptr,
                                               dA.ind,
                                               block_dim,
                                               info,
                                               dB,
                                               dB.ld,
                                               dX,
                                               dX.ld,
                                               spol,
                                               dbuffer);

        rocsparse_bsrsm_zero_pivot(handle, info, solve_pivot_gold);

        double gflop_count
            = csrsv_gflop_count(m, size_t(dA.nnzb) * block_dim * block_dim, diag) * nrhs;
        double gbyte_count = bsrsv_gbyte_count<T>(mb, dA.nnzb, block_dim) * nrhs;

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        rocsparse_int pivot = (*analysis_pivot_gold != -1 && *solve_pivot_gold != -1)
                                  ? std::min(*analysis_pivot_gold, *solve_pivot_gold)
                                  : std::max(*analysis_pivot_gold, *solve_pivot_gold);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::nnz,
                            size_t(hA.nnzb) * block_dim * block_dim,
                            display_key_t::nrhs,
                            nrhs,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::pivot,
                            pivot,
                            display_key_t::trans_A,
                            rocsparse_operation2string(trans_A),
                            display_key_t::trans_X,
                            rocsparse_operation2string(trans_X),
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

    // Clear bsrsm meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsm_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrsm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrsm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrsm_extra(const Arguments& arg) {}

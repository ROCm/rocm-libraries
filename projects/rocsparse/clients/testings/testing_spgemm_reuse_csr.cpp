/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename J, typename T>
void testing_spgemm_reuse_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle = local_handle;
    J                m      = safe_size;
    J                n      = safe_size;
    J                k      = safe_size;
    I                nnz_A  = safe_size;
    I                nnz_B  = safe_size;
    I                nnz_C  = safe_size;
    I                nnz_D  = safe_size;

    void* csr_row_ptr_A = (void*)0x4;
    void* csr_col_ind_A = (void*)0x4;
    void* csr_val_A     = (void*)0x4;
    void* csr_row_ptr_B = (void*)0x4;
    void* csr_col_ind_B = (void*)0x4;
    void* csr_val_B     = (void*)0x4;
    void* csr_row_ptr_C = (void*)0x4;
    void* csr_col_ind_C = (void*)0x4;
    void* csr_val_C     = (void*)0x4;
    void* csr_row_ptr_D = (void*)0x4;
    void* csr_col_ind_D = (void*)0x4;
    void* csr_val_D     = (void*)0x4;

    rocsparse_operation    trans_A = rocsparse_operation_none;
    rocsparse_operation    trans_B = rocsparse_operation_none;
    rocsparse_index_base   base    = rocsparse_index_base_zero;
    rocsparse_spgemm_alg   alg     = rocsparse_spgemm_alg_default;
    rocsparse_spgemm_stage stage   = rocsparse_spgemm_stage_numeric;

    // Index and data type
    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    // SpGEMM structures
    rocsparse_local_spmat local_A(
        m, k, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_val_A, itype, jtype, base, compute_type);
    rocsparse_local_spmat local_B(
        k, n, nnz_B, csr_row_ptr_B, csr_col_ind_B, csr_val_B, itype, jtype, base, compute_type);
    rocsparse_local_spmat local_C(
        m, n, nnz_C, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, compute_type);
    rocsparse_local_spmat local_D(
        m, n, nnz_D, csr_row_ptr_D, csr_col_ind_D, csr_val_D, itype, jtype, base, compute_type);

    rocsparse_spmat_descr A = local_A;
    rocsparse_spmat_descr B = local_B;
    rocsparse_spmat_descr C = local_C;
    rocsparse_spmat_descr D = local_D;

    int       nargs_to_exclude   = 4;
    const int args_to_exclude[4] = {3, 6, 12, 13};

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

#define PARAMS                                                                                \
    handle, trans_A, trans_B, alpha, A, B, beta, D, C, compute_type, alg, stage, buffer_size, \
        temp_buffer
    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################
    {
        const T* alpha       = (const T*)nullptr;
        const T* beta        = (const T*)nullptr;
        size_t*  buffer_size = (size_t*)0x4;
        void*    temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################
    {
        const T* alpha = (const T*)0x4;
        const T* beta  = (const T*)nullptr;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################
    {
        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)0x4;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################
    {
        const T* alpha = (const T*)0x4;
        const T* beta  = (const T*)0x4;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS
}

template <typename I, typename J, typename T>
void testing_spgemm_reuse_csr(const Arguments& arg)
{
    J                    M       = arg.M;
    J                    N       = arg.N;
    J                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base_A  = arg.baseA;
    rocsparse_index_base base_B  = arg.baseB;
    rocsparse_index_base base_C  = arg.baseC;
    rocsparse_index_base base_D  = arg.baseD;
    rocsparse_spgemm_alg alg     = arg.spgemm_alg;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // -99 means nullptr
    T* h_alpha_ptr = (h_alpha == (T)-99) ? nullptr : &h_alpha;
    T* h_beta_ptr  = (h_beta == (T)-99) ? nullptr : &h_beta;

    // Index and data type
    rocsparse_datatype ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Declare host matrices.
    host_csr_matrix<T, I, J> hA, hB, hD;

    // Init matrix A
    const bool                        to_int    = arg.timing ? false : true;
    static constexpr bool             full_rank = false;
    rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
    matrix_factory.init_csr(hA, M, K, base_A);

    // Init matrix B and D from rocsparse_matrix_init random.
    {
        static constexpr bool             noseed = true;
        rocsparse_matrix_factory<T, I, J> matrix_factory(
            arg, rocsparse_matrix_random, to_int, full_rank, noseed);
        matrix_factory.init_csr(hB, K, N, base_B);
        matrix_factory.init_csr(hD, M, N, base_D);
    }

    // Declare device matrices.
    device_csr_matrix<T, I, J> dA(hA);
    device_csr_matrix<T, I, J> dB(hB);
    device_csr_matrix<T, I, J> dD(hD);

    // Declare local spmat.
    rocsparse_local_spmat A(dA), B(dB), D(dD);

    device_csr_matrix<T, I, J> dC;
    dC.define(M, N, 0, base_C);
    rocsparse_local_spmat C(dC);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Compute buffer size.
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           h_alpha_ptr,
                                           A,
                                           B,
                                           h_beta_ptr,
                                           D,
                                           C,
                                           ttype,
                                           alg,
                                           rocsparse_spgemm_stage_buffer_size,
                                           &buffer_size,
                                           nullptr));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    // Compute nnz and fill row pointer for C.
    CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           h_alpha_ptr,
                                           A,
                                           B,
                                           h_beta_ptr,
                                           D,
                                           C,
                                           ttype,
                                           alg,
                                           rocsparse_spgemm_stage_nnz,
                                           &buffer_size,
                                           dbuffer));

    // Update memory.
    int64_t C_m, C_n, C_nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C, &C_m, &C_n, &C_nnz));
    dC.define(dC.m, dC.n, C_nnz, dC.base);
    CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_pointers(C, dC.ptr, dC.ind, dC.val));

    // Compute symbolic C.
    CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           h_alpha_ptr,
                                           A,
                                           B,
                                           h_beta_ptr,
                                           D,
                                           C,
                                           ttype,
                                           alg,
                                           rocsparse_spgemm_stage_symbolic,
                                           &buffer_size,
                                           dbuffer));

    // Compute symbolic C on host.
    host_csr_matrix<T, I, J> hC_gold;

    I hC_gold_nnz = 0;
    hC_gold.define(M, N, hC_gold_nnz, base_C);
    host_csrgemm_nnz<T, I, J>(M,
                              N,
                              K,
                              h_alpha_ptr,
                              hA.ptr,
                              hA.ind,
                              hB.ptr,
                              hB.ind,
                              h_beta_ptr,
                              hD.ptr,
                              hD.ind,
                              hC_gold.ptr,
                              &hC_gold_nnz,
                              hA.base,
                              hB.base,
                              hC_gold.base,
                              hD.base);
    hC_gold.define(hC_gold.m, hC_gold.n, hC_gold_nnz, hC_gold.base);

    if(arg.unit_check)
    {
        // Compute numeric C.
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spgemm_stage_numeric,
                                               &buffer_size,
                                               dbuffer));

        // Compute numeric C on host
        host_csrgemm<T, I, J>(M,
                              N,
                              K,
                              h_alpha_ptr,
                              hA.ptr,
                              hA.ind,
                              hA.val,
                              hB.ptr,
                              hB.ind,
                              hB.val,
                              h_beta_ptr,
                              hD.ptr,
                              hD.ind,
                              hD.val,
                              hC_gold.ptr,
                              hC_gold.ind,
                              hC_gold.val,
                              hA.base,
                              hB.base,
                              hC_gold.base,
                              hD.base);

        hC_gold.near_check(dC);

        // Change values in input matrices
        rocsparse_init<T>(hA.val, 1, hA.nnz, 1, arg.convert_to_int);
        rocsparse_init<T>(hB.val, 1, hB.nnz, 1, arg.convert_to_int);
        rocsparse_init<T>(hD.val, 1, hD.nnz, 1, arg.convert_to_int);

        dA.transfer_from(hA);
        dB.transfer_from(hB);
        dD.transfer_from(hD);

        // Compute numeric C.
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spgemm_stage_numeric,
                                               &buffer_size,
                                               dbuffer));

        // Compute numeric C on host
        host_csrgemm<T, I, J>(M,
                              N,
                              K,
                              h_alpha_ptr,
                              hA.ptr,
                              hA.ind,
                              hA.val,
                              hB.ptr,
                              hB.ind,
                              hB.val,
                              h_beta_ptr,
                              hD.ptr,
                              hD.ind,
                              hD.val,
                              hC_gold.ptr,
                              hC_gold.ind,
                              hC_gold.val,
                              hA.base,
                              hB.base,
                              hC_gold.base,
                              hD.base);

        hC_gold.near_check(dC);
    }

    if(arg.timing)
    {

        const double gpu_solve_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spgemm,
                                               handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spgemm_stage_numeric,
                                               &buffer_size,
                                               dbuffer);

        double gflop_count = csrgemm_gflop_count<T, I, J>(
            M, &h_alpha, hA.ptr, hA.ind, hB.ptr, &h_beta, hD.ptr, hA.base);

        double gbyte_count = csrgemm_gbyte_count<T, I, J>(
            M, N, K, hA.nnz, hB.nnz, C_nnz, hD.nnz, &h_alpha, &h_beta);

        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::nnz_A,
                            dA.nnz,
                            display_key_t::nnz_B,
                            dB.nnz,
                            display_key_t::nnz_C,
                            C_nnz,
                            display_key_t::nnz_D,
                            dD.nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                       \
    template void testing_spgemm_reuse_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spgemm_reuse_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_spgemm_reuse_csr_extra(const Arguments& arg) {}

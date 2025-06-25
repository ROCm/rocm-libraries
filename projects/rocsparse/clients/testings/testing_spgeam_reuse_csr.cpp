/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_spgeam_reuse_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle = local_handle;
    J                m      = safe_size;
    J                n      = safe_size;
    I                nnz_A  = safe_size;
    I                nnz_B  = safe_size;
    I                nnz_C  = safe_size;

    rocsparse_spgeam_descr descr = (rocsparse_spgeam_descr)0x4;

    void* csr_row_ptr_A = (void*)0x4;
    void* csr_col_ind_A = (void*)0x4;
    void* csr_val_A     = (void*)0x4;
    void* csr_row_ptr_B = (void*)0x4;
    void* csr_col_ind_B = (void*)0x4;
    void* csr_val_B     = (void*)0x4;
    void* csr_row_ptr_C = (void*)0x4;
    void* csr_col_ind_C = (void*)0x4;
    void* csr_val_C     = (void*)0x4;

    rocsparse_index_base base = rocsparse_index_base_zero;

    // Index and data type
    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    // SpGEAM structures
    rocsparse_local_spmat local_A(
        m, n, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_val_A, itype, jtype, base, compute_type);
    rocsparse_local_spmat local_B(
        m, n, nnz_B, csr_row_ptr_B, csr_col_ind_B, csr_val_B, itype, jtype, base, compute_type);
    rocsparse_local_spmat local_C(
        m, n, nnz_C, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, compute_type);

    rocsparse_spmat_descr mat_A = local_A;
    rocsparse_spmat_descr mat_B = local_B;
    rocsparse_spmat_descr mat_C = local_C;

    size_t buffer_size = 0;
    void*  temp_buffer = nullptr;

    int32_t       nargs_to_exclude_buffer_size   = 3;
    const int32_t args_to_exclude_buffer_size[3] = {4, 6, 7};

    int32_t       nargs_to_exclude_analysis   = 4;
    const int32_t args_to_exclude_analysis[4] = {4, 6, 7, 8};

    int32_t       nargs_to_exclude   = 3;
    const int32_t args_to_exclude[3] = {6, 7, 8};

    rocsparse_spgeam_stage stage = rocsparse_spgeam_stage_symbolic_analysis;
    rocsparse_error*       p_error{};
    // Analysis
    select_bad_arg_analysis(rocsparse_spgeam_buffer_size,
                            nargs_to_exclude_buffer_size,
                            args_to_exclude_buffer_size,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            &buffer_size,
                            p_error);

    select_bad_arg_analysis(rocsparse_spgeam,
                            nargs_to_exclude_analysis,
                            args_to_exclude_analysis,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            buffer_size,
                            temp_buffer,
                            p_error);

    stage = rocsparse_spgeam_stage_symbolic_compute;

    // Symbolic_Compute
    select_bad_arg_analysis(rocsparse_spgeam_buffer_size,
                            nargs_to_exclude_buffer_size,
                            args_to_exclude_buffer_size,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            &buffer_size,
                            p_error);

    select_bad_arg_analysis(rocsparse_spgeam,
                            nargs_to_exclude,
                            args_to_exclude,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            buffer_size,
                            temp_buffer,
                            p_error);

    // Numeric analysis
    stage = rocsparse_spgeam_stage_numeric_analysis;

    select_bad_arg_analysis(rocsparse_spgeam_buffer_size,
                            nargs_to_exclude_buffer_size,
                            args_to_exclude_buffer_size,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            &buffer_size,
                            p_error);

    select_bad_arg_analysis(rocsparse_spgeam,
                            nargs_to_exclude_analysis,
                            args_to_exclude_analysis,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            buffer_size,
                            temp_buffer,
                            p_error);

    // Numeric compute
    stage = rocsparse_spgeam_stage_numeric_compute;
    select_bad_arg_analysis(rocsparse_spgeam_buffer_size,
                            nargs_to_exclude_buffer_size,
                            args_to_exclude_buffer_size,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            &buffer_size,
                            p_error);

    select_bad_arg_analysis(rocsparse_spgeam,
                            nargs_to_exclude,
                            args_to_exclude,
                            handle,
                            descr,
                            mat_A,
                            mat_B,
                            mat_C,
                            stage,
                            buffer_size,
                            temp_buffer,
                            p_error);
}

template <typename I, typename J, typename T>
void testing_spgeam_reuse_csr(const Arguments& arg)
{
    J                    M       = arg.M;
    J                    N       = arg.N;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base_A  = arg.baseA;
    rocsparse_index_base base_B  = arg.baseB;
    rocsparse_index_base base_C  = arg.baseC;
    rocsparse_spgeam_alg alg     = arg.spgeam_alg;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    T* h_alpha_ptr = &h_alpha;
    T* h_beta_ptr  = &h_beta;

    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
    T* d_alpha_ptr = d_alpha;
    T* d_beta_ptr  = d_beta;

    // Index and data type
    rocsparse_datatype ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Grab stream from handle
    hipStream_t stream = handle.get_stream();

    // Declare host matrices.
    host_csr_matrix<T, I, J> hA, hB;

    const bool            to_int    = arg.timing ? false : true;
    static constexpr bool full_rank = false;

    // Init matrix A from the input rocsparse_matrix_init
    {
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        matrix_factory.init_csr(hA, M, N, base_A);
    }

    // Init matrix B from rocsparse_matrix_init random.
    {
        static constexpr bool             noseed = true;
        rocsparse_matrix_factory<T, I, J> matrix_factory(
            arg, rocsparse_matrix_random, to_int, full_rank, noseed);
        matrix_factory.init_csr(hB, M, N, base_B);
    }

    // Declare device matrices.
    device_csr_matrix<T, I, J> dA(hA);
    device_csr_matrix<T, I, J> dB(hB);

    // Declare local spmat.
    rocsparse_local_spmat mat_A(dA), mat_B(dB);

    rocsparse_spgeam_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_spgeam_descr(&descr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_A, &trans_A, sizeof(trans_A), nullptr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_B, &trans_B, sizeof(trans_B), nullptr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_datatype, &ttype, sizeof(ttype), nullptr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_compute_datatype, &ttype, sizeof(ttype), nullptr));

    // Calculate NNZ phase
    size_t buffer_size_in_bytes;
    void*  buffer;
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size(handle,
                                                       descr,
                                                       mat_A,
                                                       mat_B,
                                                       nullptr,
                                                       rocsparse_spgeam_stage_symbolic_analysis,
                                                       &buffer_size_in_bytes,
                                                       nullptr));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                           descr,
                                           mat_A,
                                           mat_B,
                                           nullptr,
                                           rocsparse_spgeam_stage_symbolic_analysis,
                                           buffer_size_in_bytes,
                                           buffer,
                                           nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

    // Ensure analysis stage is complete before grabbing C non-zero count
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    int64_t nnz_C;
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_get_output(
        handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(int64_t), nullptr));

    // Allocate and set up C
    device_csr_matrix<T, I, J> dC;
    dC.define(M, N, nnz_C, base_C);
    rocsparse_local_spmat mat_C(dC);

    // Symbolic compute phase
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size(handle,
                                                       descr,
                                                       mat_A,
                                                       mat_B,
                                                       mat_C,
                                                       rocsparse_spgeam_stage_symbolic_compute,
                                                       &buffer_size_in_bytes,
                                                       nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                           descr,
                                           mat_A,
                                           mat_B,
                                           mat_C,
                                           rocsparse_spgeam_stage_symbolic_compute,
                                           buffer_size_in_bytes,
                                           buffer,
                                           nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

    // Numeric compute phase
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size(handle,
                                                       descr,
                                                       mat_A,
                                                       mat_B,
                                                       mat_C,
                                                       rocsparse_spgeam_stage_numeric_analysis,
                                                       &buffer_size_in_bytes,
                                                       nullptr));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                           descr,
                                           mat_A,
                                           mat_B,
                                           mat_C,
                                           rocsparse_spgeam_stage_numeric_analysis,
                                           buffer_size_in_bytes,
                                           buffer,
                                           nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size(handle,
                                                       descr,
                                                       mat_A,
                                                       mat_B,
                                                       mat_C,
                                                       rocsparse_spgeam_stage_numeric_compute,
                                                       &buffer_size_in_bytes,
                                                       nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

    if(arg.unit_check)
    {
        // Compute C on host.
        host_csr_matrix<T, I, J> hC;

        I hC_nnz = 0;
        hC.define(M, N, hC_nnz, base_C);
        host_csrgeam_nnz<T, I, J>(M,
                                  N,
                                  h_alpha_ptr,
                                  hA.ptr,
                                  hA.ind,
                                  h_beta_ptr,
                                  hB.ptr,
                                  hB.ind,
                                  hC.ptr,
                                  &hC_nnz,
                                  hA.base,
                                  hB.base,
                                  hC.base);
        hC.define(hC.m, hC.n, hC_nnz, hC.base);

        host_csrgeam<T, I, J>(M,
                              N,
                              h_alpha_ptr,
                              hA.ptr,
                              hA.ind,
                              hA.val,
                              h_beta_ptr,
                              hB.ptr,
                              hB.ind,
                              hB.val,
                              hC.ptr,
                              hC.ind,
                              hC.val,
                              hA.base,
                              hB.base,
                              hC.base);

        // Compute C on host multiple times.
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_alpha,
                                                         h_alpha_ptr,
                                                         sizeof(h_alpha_ptr),
                                                         nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_beta,
                                                         h_beta_ptr,
                                                         sizeof(h_beta_ptr),
                                                         nullptr));

        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_numeric_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
        }

        hC.near_check(dC);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC pointer mode host", dC);
        }

        // Compute C on device multiple times.
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_alpha,
                                                         d_alpha_ptr,
                                                         sizeof(d_alpha_ptr),
                                                         nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_beta,
                                                         d_beta_ptr,
                                                         sizeof(d_beta_ptr),
                                                         nullptr));

        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_numeric_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
        }

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC pointer mode device", dC);
        }

        hC.near_check(dC);
    }

    if(arg.timing)
    {
        int32_t number_cold_calls = 2;
        int32_t number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_alpha,
                                                         h_alpha_ptr,
                                                         sizeof(h_alpha_ptr),
                                                         nullptr));

        CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(handle,
                                                         descr,
                                                         rocsparse_spgeam_input_scalar_beta,
                                                         h_beta_ptr,
                                                         sizeof(d_beta_ptr),
                                                         nullptr));

        // Warm up
        for(int32_t iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_numeric_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int32_t iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_numeric_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gflop_count
            = csrgeam_gflop_count<T, I, J>(hA.nnz, hB.nnz, nnz_C, h_alpha_ptr, h_beta_ptr);

        double gbyte_count
            = csrgeam_gbyte_count<T, I, J>(M, hA.nnz, hB.nnz, nnz_C, h_alpha_ptr, h_beta_ptr);

        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);

        display_timing_info(display_key_t::trans_A,
                            rocsparse_operation2string(trans_A),
                            display_key_t::trans_B,
                            rocsparse_operation2string(trans_B),
                            display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz_A,
                            dA.nnz,
                            display_key_t::nnz_B,
                            dB.nnz,
                            display_key_t::nnz_C,
                            nnz_C,
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

    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spgeam_descr(descr));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                       \
    template void testing_spgeam_reuse_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spgeam_reuse_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

//
// This test messes up stages and checks that invalid status is returned.
//
static void testing_spgeam_reuse_csr_extra_wrong_stages(const Arguments& arg)
{
    const int32_t              m       = 2;
    const int32_t              n       = 2;
    const int32_t              nnz_A   = 2;
    const int32_t              nnz_B   = 2;
    const int32_t              nnz_C   = 2;
    const float                h_alpha = 1.0f;
    const float                h_beta  = 1.0f;
    const rocsparse_index_base base_A  = rocsparse_index_base_zero;
    const rocsparse_index_base base_B  = rocsparse_index_base_zero;
    const rocsparse_index_base base_C  = rocsparse_index_base_zero;
    const rocsparse_operation  trans_A = rocsparse_operation_none;
    const rocsparse_operation  trans_B = rocsparse_operation_none;
    const rocsparse_spgeam_alg alg     = rocsparse_spgeam_alg_default;
    const rocsparse_datatype   ttype   = rocsparse_datatype_f32_r;
    // Create rocsparse handle
    rocsparse_local_handle handle;
    // Declare host matrices.
    host_csr_matrix<float> hA(m, n, nnz_A, base_A);
    host_csr_matrix<float> hB(m, n, nnz_B, base_B);
    host_csr_matrix<float> hC(m, n, nnz_C, base_C);
    hA.ptr[0] = 0;
    hA.ptr[1] = 1;
    hA.ptr[2] = 2;
    hA.ind[0] = 0;
    hA.ind[1] = 1;
    hA.val[0] = 1;
    hA.val[1] = 1;

    hB.ptr[0] = 0;
    hB.ptr[1] = 1;
    hB.ptr[2] = 2;
    hB.ind[0] = 0;
    hB.ind[1] = 1;
    hB.val[0] = 1;
    hB.val[1] = 1;

    hC.ptr[0] = 0;
    hC.ptr[1] = 1;
    hC.ptr[2] = 2;
    hC.ind[0] = 0;
    hC.ind[1] = 1;
    hC.val[0] = 2;
    hC.val[1] = 2;

    device_csr_matrix<float> dA(hA), dB(hB), dC(hC);
    rocsparse_local_spmat    A(dA), B(dB), C(dC);

    rocsparse_spgeam_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_spgeam_descr(&descr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_A, &trans_A, sizeof(trans_A), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_B, &trans_B, sizeof(trans_B), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_datatype, &ttype, sizeof(ttype), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_compute_datatype, &ttype, sizeof(ttype), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_alpha, &h_alpha, sizeof(&h_alpha), nullptr));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_beta, &h_beta, sizeof(&h_beta), nullptr));

    // Calculate NNZ phase
    const size_t              buffer_size_in_bytes = 64;
    device_dense_vector<char> buffer(buffer_size_in_bytes);

#define PARAMS(stage) handle, descr, A, B, C, stage, buffer_size_in_bytes, buffer, nullptr

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling symbolic_compute before symbolic analysis
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_compute)),
                                rocsparse_status_invalid_value);
        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    //
    // Call symbolic analysis
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_analysis)));

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling symbolic analysis twice
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_analysis)),
                                rocsparse_status_invalid_value);

        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    //
    // Call symbolic_compute
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_compute)));

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling symbolic_compute twice
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_compute)),
                                rocsparse_status_invalid_value);

        //
        // Expect an invalid status when calling compute after symbolic_compute
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_compute)),
                                rocsparse_status_invalid_value);

        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling numeric_compute before numeric_analysis
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_compute)),
                                rocsparse_status_invalid_value);
        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    //
    // Call numeric analysis
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_numeric_analysis)));

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling numeric analysis twice
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_numeric_analysis)),
                                rocsparse_status_invalid_value);

        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    //
    // Call numeric_compute
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_numeric_compute)));

    //
    // Call numeric_compute twice
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_numeric_compute)));

    {
        ROCSPARSE_DEBUG_VERBOSE_OFF;
        //
        // Expect an invalid status when calling symbolic_compute after numeric_compute
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_symbolic_compute)),
                                rocsparse_status_invalid_value);

        //
        // Expect an invalid status when calling compute after numeric_compute
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgeam(PARAMS(rocsparse_spgeam_stage_compute)),
                                rocsparse_status_invalid_value);

        ROCSPARSE_DEBUG_VERBOSE_ON;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spgeam_descr(descr));
#undef PARAMS
}

void testing_spgeam_reuse_csr_extra(const Arguments& arg)
{
    testing_spgeam_reuse_csr_extra_wrong_stages(arg);
}

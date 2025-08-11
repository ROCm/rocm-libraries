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
void testing_spgeam_csr_2_bad_arg(const Arguments& arg)
{
}

template <typename I, typename J, typename T>
void testing_spgeam_csr_2(const Arguments& arg)
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

    // Allocate and set up C
    device_csr_matrix<T, I, J> dC(M, N, 0, base_C);

    // Declare local spmat.
    rocsparse_local_spmat mat_A(dA), mat_B(dB), mat_C(dC);

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
                                                       mat_C,
                                                       rocsparse_spgeam_stage_analysis,
                                                       &buffer_size_in_bytes,
                                                       nullptr));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                           descr,
                                           mat_A,
                                           mat_B,
                                           mat_C,
                                           rocsparse_spgeam_stage_analysis,
                                           buffer_size_in_bytes,
                                           buffer,
                                           nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

    // Ensure analysis stage is complete before grabbing C non-zero count
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    int64_t nnz_C;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_nnz(mat_C, &nnz_C));

    // allocate and set up C
    dC.define(M, N, nnz_C, base_C);

    // Set C pointers
    CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_pointers(mat_C, dC.ptr, dC.ind, dC.val));

    // Compute phase
    CHECK_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size(handle,
                                                       descr,
                                                       mat_A,
                                                       mat_B,
                                                       mat_C,
                                                       rocsparse_spgeam_stage_compute,
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

        for(int i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
            hC.near_check(dC);
        }

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

        for(int i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
            hC.near_check(dC);
        }

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC pointer mode device", dC);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

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

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   nullptr));
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgeam(handle,
                                                   descr,
                                                   mat_A,
                                                   mat_B,
                                                   mat_C,
                                                   rocsparse_spgeam_stage_compute,
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                   \
    template void testing_spgeam_csr_2_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spgeam_csr_2<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_spgeam_csr_2_extra(const Arguments& arg) {}

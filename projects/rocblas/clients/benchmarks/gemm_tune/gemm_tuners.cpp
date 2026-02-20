/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "gemm_tuners.hpp"

/* COMMON */
template <typename Tc>
GEMMTunerBase<Tc>::GEMMTunerBase(const Arguments& arg)
    : m_handle(arg)
    , m_device(arg.devices)
    , m_cold_iters(arg.cold_iters)
    , m_iters(arg.iters)
    , m_trans_A(char2rocblas_operation(arg.transA))
    , m_trans_B(char2rocblas_operation(arg.transB))
    , m_input_type(arg.a_type)
    , m_output_type(arg.c_type)
    , m_compute_type(arg.compute_type)
    , m_alpha(arg.get_alpha<Tc>())
    , m_beta(arg.get_beta<Tc>())
    , m_M(arg.M)
    , m_N(arg.N)
    , m_K(arg.K)
    , m_lda(arg.lda)
    , m_ldb(arg.ldb)
    , m_ldc(arg.ldc)
{
    // adjust dimension for GEMM routines
    int K_clip = std::max(m_K, 1);
    m_A_row    = (m_trans_A == rocblas_operation_none) ? m_M : K_clip;
    m_A_col    = (m_trans_A == rocblas_operation_none) ? K_clip : m_M;
    m_B_row    = (m_trans_B == rocblas_operation_none) ? K_clip : m_N;
    m_B_col    = (m_trans_B == rocblas_operation_none) ? m_N : K_clip;

    if(m_lda < m_A_row)
    {
        rocblas_cout << "rocblas-gemm-tune INFO: lda < min_lda, set lda = " << m_A_row << std::endl;
        m_lda = m_A_row;
    }
    if(m_ldb < m_B_row)
    {
        rocblas_cout << "rocblas-gemm-tune INFO: ldb < min_ldb, set ldb = " << m_B_row << std::endl;
        m_ldb = m_B_row;
    }
    if(m_ldc < m_M)
    {
        rocblas_cout << "rocblas-gemm-tune INFO: ldc < min_ldc, set ldc = " << m_M << std::endl;
        m_ldc = m_M;
    }
}

template <typename Tc>
int GEMMTunerBase<Tc>::get_best_solution()
{
    CHECK_HIP_ERROR(hipSetDevice(m_device));

    // Get all solutions
    rocblas_int n_solutions;
    CHECK_ROCBLAS_ERROR(get_solutions(NULL, &n_solutions));

    std::vector<rocblas_int> solutions(n_solutions);
    CHECK_ROCBLAS_ERROR(get_solutions(solutions.data(), &n_solutions));

    // Benchmark each and return best
    double         best_time = std::numeric_limits<double>::max();
    rocblas_int    best_sol  = -1;
    rocblas_status status;

    for(auto sol : solutions)
    {
        // warmup
        for(rocblas_int c = 0; c < m_cold_iters; ++c)
        {
            CHECK_ROCBLAS_ERROR(run_with_solution(sol));
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(m_handle, &stream));
        double time = get_time_us_sync(stream); // in microseconds

        // timing loop
        for(rocblas_int c = 0; c < m_iters; ++c)
        {
            CHECK_ROCBLAS_ERROR(run_with_solution(sol));
        }
        time = get_time_us_sync(stream) - time;

        // track winner
        double avg_time = m_iters ? (time / m_iters) : 0;
        if(avg_time < best_time)
        {
            best_sol  = sol;
            best_time = avg_time;
        }
    }

    return best_sol;
}

/* GEMM Ex */
template <typename Ti, typename To, typename Tc>
GEMMTunerEx<Ti, To, Tc>::GEMMTunerEx(const Arguments& arg)
    : GEMMTunerBase<Tc>(arg)
    , m_dA(this->m_A_row, this->m_A_col, this->m_lda)
    , m_dB(this->m_B_row, this->m_B_col, this->m_ldb)
    , m_dC(this->m_M, this->m_N, this->m_ldc)
{
    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(m_dA.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dB.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dC.memcheck());

    // Allocate host memory
    host_matrix<Ti> hA(this->m_A_row, this->m_A_col, this->m_lda);
    host_matrix<Ti> hB(this->m_B_row, this->m_B_col, this->m_ldb);
    host_matrix<To> hC(this->m_M, this->m_N, this->m_ldc);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());

    // Initialize data on host memory
    // We don't care about the result, but the GEMM input data affects the benchmarks
    rocblas_init_matrix<Ti>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<Ti>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(m_dA.transfer_from(hA));
    CHECK_HIP_ERROR(m_dB.transfer_from(hB));
    CHECK_HIP_ERROR(m_dC.transfer_from(hC));
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerEx<Ti, To, Tc>::get_solutions(rocblas_int* solution_list, rocblas_int* size)
{
    return rocblas_gemm_ex_get_solutions(this->m_handle,
                                         this->m_trans_A,
                                         this->m_trans_B,
                                         this->m_M,
                                         this->m_N,
                                         this->m_K,
                                         &this->m_alpha,
                                         m_dA,
                                         this->m_input_type,
                                         this->m_lda,
                                         m_dB,
                                         this->m_input_type,
                                         this->m_ldb,
                                         &this->m_beta,
                                         m_dC,
                                         this->m_output_type,
                                         this->m_ldc,
                                         m_dC,
                                         this->m_output_type,
                                         this->m_ldc,
                                         this->m_compute_type,
                                         rocblas_gemm_algo_solution_index,
                                         rocblas_gemm_flags_none,
                                         solution_list,
                                         size);
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerEx<Ti, To, Tc>::run_with_solution(int solution_idx)
{
    return rocblas_gemm_ex(this->m_handle,
                           this->m_trans_A,
                           this->m_trans_B,
                           this->m_M,
                           this->m_N,
                           this->m_K,
                           &this->m_alpha,
                           m_dA,
                           this->m_input_type,
                           this->m_lda,
                           m_dB,
                           this->m_input_type,
                           this->m_ldb,
                           &this->m_beta,
                           m_dC,
                           this->m_output_type,
                           this->m_ldc,
                           m_dC,
                           this->m_output_type,
                           this->m_ldc,
                           this->m_compute_type,
                           rocblas_gemm_algo_solution_index,
                           solution_idx,
                           rocblas_gemm_flags_none);
}

/* GEMM Batched Ex */
template <typename Ti, typename To, typename Tc>
GEMMTunerBatchedEx<Ti, To, Tc>::GEMMTunerBatchedEx(const Arguments& arg)
    : GEMMTunerBase<Tc>(arg)
    , m_batch_count(arg.batch_count)
    , m_dA(this->m_A_row, this->m_A_col, this->m_lda, m_batch_count)
    , m_dB(this->m_B_row, this->m_B_col, this->m_ldb, m_batch_count)
    , m_dC(this->m_M, this->m_N, this->m_ldc, m_batch_count)
{
    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(m_dA.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dB.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dC.memcheck());

    // Allocate host memory
    host_batch_matrix<Ti> hA(this->m_A_row, this->m_A_col, this->m_lda, m_batch_count);
    host_batch_matrix<Ti> hB(this->m_B_row, this->m_B_col, this->m_ldb, m_batch_count);
    host_batch_matrix<To> hC(this->m_M, this->m_N, this->m_ldc, m_batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix<Ti>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<Ti>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(m_dA.transfer_from(hA));
    CHECK_HIP_ERROR(m_dB.transfer_from(hB));
    CHECK_HIP_ERROR(m_dC.transfer_from(hC));
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerBatchedEx<Ti, To, Tc>::get_solutions(rocblas_int* solution_list,
                                                             rocblas_int* size)
{
    return rocblas_gemm_batched_ex_get_solutions(this->m_handle,
                                                 this->m_trans_A,
                                                 this->m_trans_B,
                                                 this->m_M,
                                                 this->m_N,
                                                 this->m_K,
                                                 &this->m_alpha,
                                                 m_dA.ptr_on_device(),
                                                 this->m_input_type,
                                                 this->m_lda,
                                                 m_dB.ptr_on_device(),
                                                 this->m_input_type,
                                                 this->m_ldb,
                                                 &this->m_beta,
                                                 m_dC.ptr_on_device(),
                                                 this->m_output_type,
                                                 this->m_ldc,
                                                 m_dC.ptr_on_device(),
                                                 this->m_output_type,
                                                 this->m_ldc,
                                                 m_batch_count,
                                                 this->m_compute_type,
                                                 rocblas_gemm_algo_solution_index,
                                                 rocblas_gemm_flags_none,
                                                 solution_list,
                                                 size);
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerBatchedEx<Ti, To, Tc>::run_with_solution(int solution_idx)
{
    return rocblas_gemm_batched_ex(this->m_handle,
                                   this->m_trans_A,
                                   this->m_trans_B,
                                   this->m_M,
                                   this->m_N,
                                   this->m_K,
                                   &this->m_alpha,
                                   m_dA.ptr_on_device(),
                                   this->m_input_type,
                                   this->m_lda,
                                   m_dB.ptr_on_device(),
                                   this->m_input_type,
                                   this->m_ldb,
                                   &this->m_beta,
                                   m_dC.ptr_on_device(),
                                   this->m_output_type,
                                   this->m_ldc,
                                   m_dC.ptr_on_device(),
                                   this->m_output_type,
                                   this->m_ldc,
                                   m_batch_count,
                                   this->m_compute_type,
                                   rocblas_gemm_algo_solution_index,
                                   solution_idx,
                                   rocblas_gemm_flags_none);
}

/* GEMM Strided Batched Ex */
template <typename Tc>
GEMMTunerStridedBase<Tc>::GEMMTunerStridedBase(const Arguments& arg)
    : GEMMTunerBase<Tc>(arg)
    , m_stride_a(arg.stride_a)
    , m_stride_b(arg.stride_b)
    , m_stride_c(arg.stride_c)
{
    // adjust stride
    rocblas_int min_stride_c = this->m_ldc * this->m_N;
    if(m_stride_c < min_stride_c)
    {
        rocblas_cout << "rocblas-gemm-tune INFO: stride_c < min_stride_c, set stride_c = "
                     << min_stride_c << std::endl;
        m_stride_c = min_stride_c;
    }
}

template <typename Ti, typename To, typename Tc>
GEMMTunerStridedBatchedEx<Ti, To, Tc>::GEMMTunerStridedBatchedEx(const Arguments& arg)
    : GEMMTunerStridedBase<Tc>(arg)
    , m_batch_count(arg.batch_count)
    , m_dA(this->m_A_row, this->m_A_col, this->m_lda, this->m_stride_a, m_batch_count)
    , m_dB(this->m_B_row, this->m_B_col, this->m_ldb, this->m_stride_b, m_batch_count)
    , m_dC(this->m_M, this->m_N, this->m_ldc, this->m_stride_c, m_batch_count)
{
    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(m_dA.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dB.memcheck());
    CHECK_DEVICE_ALLOCATION(m_dC.memcheck());

    // Allocate host memory
    host_strided_batch_matrix<Ti> hA(
        this->m_A_row, this->m_A_col, this->m_lda, this->m_stride_a, m_batch_count);
    host_strided_batch_matrix<Ti> hB(
        this->m_B_row, this->m_B_col, this->m_ldb, this->m_stride_b, m_batch_count);
    host_strided_batch_matrix<To> hC(
        this->m_M, this->m_N, this->m_ldc, this->m_stride_c, m_batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix<Ti>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<Ti>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(m_dA.transfer_from(hA));
    CHECK_HIP_ERROR(m_dB.transfer_from(hB));
    CHECK_HIP_ERROR(m_dC.transfer_from(hC));
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerStridedBatchedEx<Ti, To, Tc>::get_solutions(rocblas_int* solution_list,
                                                                    rocblas_int* size)
{
    return rocblas_gemm_strided_batched_ex_get_solutions(this->m_handle,
                                                         this->m_trans_A,
                                                         this->m_trans_B,
                                                         this->m_M,
                                                         this->m_N,
                                                         this->m_K,
                                                         &this->m_alpha,
                                                         m_dA,
                                                         this->m_input_type,
                                                         this->m_lda,
                                                         this->m_stride_a,
                                                         m_dB,
                                                         this->m_input_type,
                                                         this->m_ldb,
                                                         this->m_stride_b,
                                                         &this->m_beta,
                                                         m_dC,
                                                         this->m_output_type,
                                                         this->m_ldc,
                                                         this->m_stride_c,
                                                         m_dC,
                                                         this->m_output_type,
                                                         this->m_ldc,
                                                         this->m_stride_c,
                                                         m_batch_count,
                                                         this->m_compute_type,
                                                         rocblas_gemm_algo_solution_index,
                                                         rocblas_gemm_flags_none,
                                                         solution_list,
                                                         size);
}

template <typename Ti, typename To, typename Tc>
rocblas_status GEMMTunerStridedBatchedEx<Ti, To, Tc>::run_with_solution(int solution_idx)
{
    return rocblas_gemm_strided_batched_ex(this->m_handle,
                                           this->m_trans_A,
                                           this->m_trans_B,
                                           this->m_M,
                                           this->m_N,
                                           this->m_K,
                                           &this->m_alpha,
                                           m_dA,
                                           this->m_input_type,
                                           this->m_lda,
                                           this->m_stride_a,
                                           m_dB,
                                           this->m_input_type,
                                           this->m_ldb,
                                           this->m_stride_b,
                                           &this->m_beta,
                                           m_dC,
                                           this->m_output_type,
                                           this->m_ldc,
                                           this->m_stride_c,
                                           m_dC,
                                           this->m_output_type,
                                           this->m_ldc,
                                           this->m_stride_c,
                                           m_batch_count,
                                           this->m_compute_type,
                                           rocblas_gemm_algo_solution_index,
                                           solution_idx,
                                           rocblas_gemm_flags_none);
}

#define TEMPLATE_CLASS_SINGLE(C)             \
    template class C<int32_t>;               \
    template class C<rocblas_half>;          \
    template class C<rocblas_bfloat16>;      \
    template class C<float>;                 \
    template class C<double>;                \
    template class C<rocblas_float_complex>; \
    template class C<rocblas_double_complex>;

#define TEMPLATE_CLASS_MULTI(C)                          \
    template class C<int8_t, int32_t, int32_t>;          \
    template class C<rocblas_half, float, float>;        \
    template class C<rocblas_bfloat16, float, float>;    \
    template class C<rocblas_half, rocblas_half, float>; \
    template class C<rocblas_bfloat16, rocblas_bfloat16, float>;

TEMPLATE_CLASS_SINGLE(GEMMTunerBase);
TEMPLATE_CLASS_SINGLE(GEMMTunerEx);
TEMPLATE_CLASS_SINGLE(GEMMTunerBatchedEx);
TEMPLATE_CLASS_SINGLE(GEMMTunerStridedBase);
TEMPLATE_CLASS_SINGLE(GEMMTunerStridedBatchedEx);

TEMPLATE_CLASS_MULTI(GEMMTunerEx);
TEMPLATE_CLASS_MULTI(GEMMTunerBatchedEx);
TEMPLATE_CLASS_MULTI(GEMMTunerStridedBatchedEx);

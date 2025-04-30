/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#define ROCBLAS_BETA_FEATURES_API
#include "../../library/include/internal/rocblas_float8.h" // only to set the bias mode, moved it later

#include "frequency_monitor.hpp"
#include "testing_common.hpp"

#ifdef WIN32
#include <stdlib.h>
#define setenv(A, B, C) _putenv_s(A, B)
#endif

/*
 *  Pseudo random number generator
 *      Works only when T is f32 or f16
 */
template <typename T,
          std::enable_if_t<(std::is_same<float, T>{} || std::is_same<rocblas_half, T>{}), int> = 0>
ROCBLAS_KERNEL_ILF uint32_t prand_generator_test(int64_t tid, uint32_t seed, T val)
{
    //Previous PRNG implementation
    //-----------------------------
    //float    ax  = (float)x; // can be other types for simulated codes
    //uint32_t t   = reinterpret_cast<uint32_t&>(ax);
    //t            = (t << 27) | (t >> 5);
    //uint32_t rng = (t * 0x42fe83a3) ^ 0xdfc231fd ^ (tid * 0x1791762503) ^ seed;

    //PRNG from TF-SM :
    //------------------
    //uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    //if(sizeof(x)==4)
    //    drop_bits ^= x>>16;
    //drop_bits = ((drop_bits & 31)<<11) | (drop_bits>>5);
    //drop_bits *= 0x7000149;
    //uint32_t rng = (drop_bits ^ 0x13371337 ^ (i*229791) ^ seed);

    typedef typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type IT;
    IT       x         = reinterpret_cast<IT&>(val);
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(T) == 4)
        drop_bits ^= x >> 16;
    drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (tid * 229791) ^ seed);
#ifdef PRINT_PRNG
    printf("%u , ", rng);
#endif
    return rng;
}

// NOTE: if T is not f32 or f16, we don't need down-coversion
// Therefore, we will not use SR for that. We are returning 0 as RND for those cases

template <typename T,
          std::enable_if_t<!(std::is_same<float, T>{} || std::is_same<rocblas_half, T>{}), int> = 0>
ROCBLAS_KERNEL_ILF uint32_t prand_generator_test(int64_t tid, uint32_t seed, T val)
{
    return 0;
}

/***************************************************************************************
    Quantization: New single matrix conversion routine
        TiA->TcA->TgA : we keep three types for simulated cases, like: f16->f8->f16
****************************************************************************************/
template <typename TiA, // inputAtype
          typename TcA, // computeAtype
          typename TgA, // Tensile gemm's inputA type
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          char TRANS_A,
          bool stochastic_rounding>
__attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N)))
ROCBLAS_KERNEL(DIM_M* DIM_N) general_conversion_kernel_test(rocblas_int    M,
                                                            rocblas_int    N,
                                                            const TiA*     dA_array,
                                                            TgA*           dA_array_new,
                                                            rocblas_int    lda,
                                                            rocblas_int    lda_new,
                                                            rocblas_stride stride_a,
                                                            rocblas_int    batch_count,
                                                            uint32_t       seedA)
{
    int x = BLK_M * blockIdx.x + threadIdx.x;
    int y = BLK_N * blockIdx.y + threadIdx.y;

    int blz = blockIdx.z; // block's matrix in the batch

    auto* dA = dA_array + blz * stride_a; //load_ptr_batch(dA_array, blz, 0, stride_a);

    // conversion
    for(int m = 0; m < BLK_M; m += DIM_M)
    {
        for(int n = 0; n < BLK_N; n += DIM_N)
        {
            int i = x + m;
            int j = y + n;

            if(i < M && j < N)
            {
                if(TRANS_A == 'N')
                {
                    int64_t  gid = i + j * size_t(lda);
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator_test<TiA>(gid, seedA, dA[i + j * size_t(lda)]);

                    dA_array_new[i + j * size_t(lda_new)]
                        = TgA(explicit_downcast<TcA, TiA, stochastic_rounding>(
                            dA[i + j * size_t(lda)], rng));
                }
                else if(TRANS_A == 'T')
                {
                    int64_t  gid = i * size_t(lda) + j;
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator_test<TiA>(gid, seedA, dA[i * size_t(lda) + j]);

                    dA_array_new[i * size_t(lda_new) + j] = TgA(
                        explicit_downcast<TcA, TiA, stochastic_rounding>(dA[i * lda + j], rng));
                }
                else if(TRANS_A == 'C')
                {
                    int64_t  gid = i * size_t(lda) + j;
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator_test<TiA>(
                            gid, seedA, conjugate(dA[i * size_t(lda) + j]));

                    dA_array_new[i * size_t(lda_new) + j]
                        = TgA(explicit_downcast<TcA, TiA, stochastic_rounding>(
                            conjugate(dA[i * size_t(lda) + j]), rng));
                }
            }
        }
    }
}

/* ============================================================================================ */
template <typename TiA, typename TiB, typename To, typename Tc>
void testing_gemm_ex3_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_ex3_fn
            = arg.api & c_API_FORTRAN ? rocblas_gemm_ex3_fortran : rocblas_gemm_ex3;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 101;

        const rocblas_int lda = 101;
        const rocblas_int ldb = 101;
        const rocblas_int ldc = 101;
        const rocblas_int ldd = 101;

        const rocblas_datatype    a_type                 = arg.a_type;
        const rocblas_datatype    b_type                 = arg.b_type;
        const rocblas_datatype    c_type                 = arg.c_type;
        const rocblas_datatype    d_type                 = arg.d_type;
        const rocblas_computetype composite_compute_type = arg.composite_compute_type;

        DEVICE_MEMCHECK(device_vector<float>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<float>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<float>, zero_d, (1));

        const float alpha_h(1), beta_h(1), zero_h(0);

        const float* alpha = &alpha_h;
        const float* beta  = &beta_h;
        const float* zero  = &zero_h;
        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

        rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
        rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
        rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

        int32_t     solution_index = 0;
        rocblas_int flags          = 0;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        DEVICE_MEMCHECK(device_matrix<TiA>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<TiB>, dB, (B_row, B_col, ldb));
        DEVICE_MEMCHECK(device_matrix<To>, dC, (M, N, ldc));
        DEVICE_MEMCHECK(device_matrix<To>, dD, (M, N, ldd));

        // host
        HOST_MEMCHECK(host_matrix<To>, hC, (M, N, ldc));
        rocblas_seedrand();
        rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
        dC.transfer_from(hC);

        if(rocblas_handle(handle)->getArch() < 942 || rocblas_handle(handle)->getArch() >= 1000)
        {
            // check for invalid arch
            EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                      transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      K,
                                                      alpha,
                                                      dA,
                                                      a_type,
                                                      lda,
                                                      dB,
                                                      b_type,
                                                      ldb,
                                                      beta,
                                                      dC,
                                                      c_type,
                                                      ldc,
                                                      dD,
                                                      d_type,
                                                      ldd,
                                                      composite_compute_type,
                                                      algo,
                                                      solution_index,
                                                      flags),
                                  rocblas_status_arch_mismatch);

            return;
        }

        // check for invalid enum
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  (rocblas_operation)rocblas_side_both,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  (rocblas_operation)rocblas_side_both,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_value);

        // check for invalid size
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  -1,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  -1,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  -1,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        // check for invalid leading dimension
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  rocblas_operation_none,
                                                  rocblas_operation_none,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  M - 1,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  rocblas_operation_none,
                                                  rocblas_operation_none,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  K - 1,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  rocblas_operation_transpose,
                                                  rocblas_operation_transpose,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  K - 1,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  rocblas_operation_transpose,
                                                  rocblas_operation_transpose,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  N - 1,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  M - 1,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  M - 1,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        // check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(nullptr,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  dA,
                                                  a_type,
                                                  lda,
                                                  dB,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dC, // aliased C
                                                  d_type,
                                                  ldc + 1,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);

        // If M==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  0,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  0,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  c_type,
                                                  ldc,
                                                  nullptr,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_success);

        // If alpha==0 then A, B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  zero,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_success);

        // the following tests still output to D

        // If K==0, then alpha, A and B can both be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  a_type,
                                                  lda,
                                                  nullptr,
                                                  b_type,
                                                  ldb,
                                                  beta,
                                                  dC,
                                                  c_type,
                                                  ldc,
                                                  dD,
                                                  d_type,
                                                  ldd,
                                                  composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_success);

        // alpha==0 && beta==1 must still copy C to D so no quick return

        //TODO
        // // If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
        // EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle, transA, transB, M, N, K, zero,
        // nullptr, a_type, lda, nullptr, b_type, ldb, zero, nullptr, c_type, ldc,
        // dD, d_type, ldd, composite_compute_type, algo, solution_index, flags), rocblas_status_success);
    }
}

template <typename TiA, typename TiB, typename To, typename TcA, typename TcB, typename Tacc>
rocblas_status call_trusted_gemm_f8(rocblas_handle    handle,
                                    rocblas_operation transA,
                                    rocblas_operation transB,
                                    rocblas_int       M,
                                    rocblas_int       N,
                                    rocblas_int       K,
                                    Tacc              alpha,
                                    const void* dA, // device_ptr to reuse in quantization if needed
                                    const TiA*  A,
                                    rocblas_int lda,
                                    const void* dB, // device_pte to reuse in quantization if needed
                                    const TiB*  B,
                                    rocblas_int ldb,
                                    Tacc        beta,
                                    const To*   C,
                                    rocblas_int ldc,
                                    To*         D, // host ptr
                                    rocblas_int ldd,
                                    bool        stochastic_rounding)
{
    hipStream_t stream       = handle->get_stream();
    bool        TiA_is_final = std::is_same<TiA, TcA>{};
    bool        TiB_is_final = std::is_same<TiB, TcB>{};

    // if To is F8 types, we need output quantization
    bool To_is_final = !(std::is_same<To, rocblas_f8>{} || std::is_same<To, rocblas_bf8>{});

    rocblas_int size_a, size_b, size_d;

    auto A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
    auto A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
    auto B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
    auto B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

    size_a = M * K;
    size_b = K * N;
    size_d = M * N;

    rocblas_int lda_new = transA == rocblas_operation_none ? M : K;
    rocblas_int ldb_new = transB == rocblas_operation_none ? K : N;
    rocblas_int ldd_new = M;

    RETURN_IF_MEM_ERROR(
        device_matrix<TcA>, dA_new, (TiA_is_final ? 0 : A_row, TiA_is_final ? 0 : A_col, lda_new));
    RETURN_IF_MEM_ERROR(
        device_matrix<TcB>, dB_new, (TiB_is_final ? 0 : B_row, TiB_is_final ? 0 : B_col, ldb_new));

    RETURN_IF_MEM_ERROR(
        host_matrix<TcA>, hA_new, (TiA_is_final ? 0 : A_row, TiA_is_final ? 0 : A_col, lda_new));
    RETURN_IF_MEM_ERROR(
        host_matrix<TcB>, hB_new, (TiB_is_final ? 0 : B_row, TiB_is_final ? 0 : B_col, ldb_new));
    RETURN_IF_MEM_ERROR(
        host_matrix<Tacc>, hD_new, (To_is_final ? 0 : M, To_is_final ? 0 : N, ldd_new));
    // temp to copyback from device
    RETURN_IF_MEM_ERROR(
        host_matrix<To>, hDo_new, (To_is_final ? 0 : M, To_is_final ? 0 : N, ldd_new));

    if(!To_is_final)
    {
        auto* D_ptr = hD_new[0];
        // auto* C_ptr = C[0];
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                D_ptr[i + j * ldd_new] = static_cast<float>(C[i + j * ldc]);
    }

    const int dim_m = 16;
    const int dim_n = 16;
    const int blk_m = 32;
    const int blk_n = 32;
    const int blk_k = 32;

    uint32_t seedA = 0, seedB = 0, seedD = 0;

    if(stochastic_rounding)
    {
        const char* sr_seed_string_a = std::getenv("SR_SEED_A");
        const char* sr_seed_string_b = std::getenv("SR_SEED_B");
        const char* sr_seed_string_d = std::getenv("SR_SEED_D");
        if(sr_seed_string_a != NULL)
        {
            seedA = std::strtol(sr_seed_string_a, NULL, 10);
        }
        if(sr_seed_string_b != NULL)
        {
            seedB = std::strtol(sr_seed_string_b, NULL, 10);
        }
        if(sr_seed_string_d != NULL)
        {
            seedD = std::strtol(sr_seed_string_d, NULL, 10);
        }
    }

    if(!TiA_is_final)
    {
        //quantization

        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(((M - 1) / blk_m) + 1, ((K - 1) / blk_k) + 1, 1);

        if(rocblas_operation_none == transA)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'N',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'N',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
        }
        else if(rocblas_operation_transpose == transA)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'T',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'T',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
        }
        else if(rocblas_operation_conjugate_transpose == transA)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'C',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiA,
                                                                      TcA,
                                                                      TcA, // Tensile gemm's TiA
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_m,
                                                                      blk_k,
                                                                      'C',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      M,
                                      K,
                                      (const TiA*)dA,
                                      dA_new,
                                      lda,
                                      lda_new,
                                      1,
                                      1,
                                      seedA);
        }

        RETURN_IF_HIP_ERROR(hA_new.transfer_from(dA_new));
    }

    if(!TiB_is_final)
    {
        //quantization

        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(((K - 1) / blk_k) + 1, ((N - 1) / blk_n) + 1, 1);

        if(rocblas_operation_none == transB)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'N',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'N',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
        }
        else if(rocblas_operation_transpose == transB)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'T',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'T',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
        }
        else if(rocblas_operation_conjugate_transpose == transB)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'C',
                                                                      true>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<TiB,
                                                                      TcB,
                                                                      TcB, // Tensile gemm's TiB
                                                                      dim_m,
                                                                      dim_n,
                                                                      blk_k,
                                                                      blk_n,
                                                                      'C',
                                                                      false>),
                                      dimGrid,
                                      dimBlock,
                                      0,
                                      stream,
                                      K,
                                      N,
                                      (const TiB*)dB,
                                      dB_new,
                                      ldb,
                                      ldb_new,
                                      1,
                                      1,
                                      seedB);
        }

        RETURN_IF_HIP_ERROR(hB_new.transfer_from(dB_new));
    }

    /*
 *  Note: F8 tensile kernels does output quantization when needed. We can't do it inside reference kernel
 *  since we have to use h/w intrinsic for CVT/CVT_SR.
 *  Therefore, we are doing output quantization in seperate device kernel which effectively the same. We are
 *  not concern about the performance of reference kernel (only used in rocblas-test)
 */

    if(!TiA_is_final && !TiB_is_final)
    {
        if(!To_is_final)
        {
            f8_to_cblas_sgemm<TcA, TcB, Tacc, Tacc>(transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    alpha,
                                                    hA_new,
                                                    lda_new,
                                                    hB_new,
                                                    ldb_new,
                                                    beta,
                                                    hD_new,
                                                    ldd_new);
        }
        else // To_is_final
        {
            f8_to_cblas_sgemm<TcA, TcB, To, Tacc>(
                transA, transB, M, N, K, alpha, hA_new, lda_new, hB_new, ldb_new, beta, D, ldd);
        }
    }
    else if(!TiA_is_final && TiB_is_final)
    {
        if(!To_is_final)
        {
            f8_to_cblas_sgemm<TcA, TiB, Tacc, Tacc>(
                transA, transB, M, N, K, alpha, hA_new, lda_new, B, ldb, beta, hD_new, ldd_new);
        }
        else // To_is_final
        {
            f8_to_cblas_sgemm<TcA, TiB, To, Tacc>(
                transA, transB, M, N, K, alpha, hA_new, lda_new, B, ldb, beta, D, ldd);
        }
    }
    else if(TiA_is_final && !TiB_is_final)
    {
        if(!To_is_final)
        {
            f8_to_cblas_sgemm<TiA, TcB, Tacc, Tacc>(
                transA, transB, M, N, K, alpha, A, lda, hB_new, ldb_new, beta, hD_new, ldd_new);
        }
        else // To_is_final
        {
            f8_to_cblas_sgemm<TiA, TcB, To, Tacc>(
                transA, transB, M, N, K, alpha, A, lda, hB_new, ldb_new, beta, D, ldd);
        }
    }
    else
    {
        if(!To_is_final)
        {
            f8_to_cblas_sgemm<TiA, TiB, Tacc, Tacc>(
                transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, hD_new, ldd_new);
        }
        else // To_is_final
        {
            f8_to_cblas_sgemm<TiA, TiB, To, Tacc>(
                transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
        }
    }

    /*
 *          Output Quantization
 *      =====================================
 */
    if(!To_is_final)
    {
        RETURN_IF_MEM_ERROR(device_vector<Tacc>, dD_Qi, (size_d));
        RETURN_IF_MEM_ERROR(device_vector<To>, dD_Qo, (size_d));

        // from hD_new to dD_Qi
        // CHECK_HIP_ERROR(dD_Qi.transfer_from(hD_new));
        RETURN_IF_HIP_ERROR(hipMemcpy(dD_Qi, hD_new, sizeof(Tacc) * size_d, hipMemcpyHostToDevice));

        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 32;
        const int blk_n = 32;
        dim3      dimBlock(dim_m, dim_n, 1);
        dim3      dimGrid(((M - 1) / blk_m) + 1, ((N - 1) / blk_n) + 1, 1);

        if(stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<Tacc,
                                                                  To,
                                                                  To,
                                                                  dim_m,
                                                                  dim_n,
                                                                  blk_m,
                                                                  blk_n,
                                                                  'N',
                                                                  true>),
                                  dimGrid,
                                  dimBlock,
                                  0,
                                  stream,
                                  M,
                                  N,
                                  (const Tacc*)dD_Qi,
                                  dD_Qo,
                                  ldd_new,
                                  ldd_new,
                                  1,
                                  1,
                                  seedD);
        else
            ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel_test<Tacc,
                                                                  To,
                                                                  To,
                                                                  dim_m,
                                                                  dim_n,
                                                                  blk_m,
                                                                  blk_n,
                                                                  'N',
                                                                  false>),
                                  dimGrid,
                                  dimBlock,
                                  0,
                                  stream,
                                  M,
                                  N,
                                  (const Tacc*)dD_Qi,
                                  dD_Qo,
                                  ldd_new,
                                  ldd_new,
                                  1,
                                  1,
                                  seedD);

        // hD0_new and dD_qo is same size vector
        RETURN_IF_HIP_ERROR(hipMemcpy(hDo_new, dD_Qo, sizeof(To) * size_d, hipMemcpyDeviceToHost));

        // copy hD0_new to D, both are in host
        for(rocblas_int batch_index = 0; batch_index < hDo_new.batch_count(); ++batch_index)
        {
            auto* Do = hDo_new[batch_index];
            // auto* hD = D[batch_index];
            for(rocblas_int j = 0; j < N; j++)
            {
                // consecutive in M_dim
                for(rocblas_int i = 0; i < M; i++)
                {
                    D[i + j * ldd] = Do[i + j * ldd_new];
                }
            }
        }
    }

    return rocblas_status_success;
}

template <typename TiA, typename TiB, typename To, typename Tc>
void testing_gemm_ex3(const Arguments& arg)
{
    auto rocblas_gemm_ex3_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemm_ex3_fortran : rocblas_gemm_ex3;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool stochastic_rounding = flags & rocblas_gemm_flags_stochastic_rounding;

    if(stochastic_rounding)
    {
        std::random_device                      rd;
        std::mt19937                            gen(rd());
        std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
        uint32_t                                seedA = 0, seedB = 0;

        seedA = distribution(gen);
        seedB = distribution(gen);

        int setenv_status;

        setenv_status = setenv("SR_SEED_A", std::to_string(seedA).c_str(), false);

#ifdef GOOGLE_TEST
        ASSERT_EQ(setenv_status, 0);
#endif

        setenv_status = setenv("SR_SEED_B", std::to_string(seedB).c_str(), false);

#ifdef GOOGLE_TEST
        ASSERT_EQ(setenv_status, 0);
#endif
    }

    bool alpha_isnan = arg.alpha_isnan<Tc>();
    bool beta_isnan  = arg.beta_isnan<Tc>();

    if(!std::is_same<To, float>{} && !std::is_same<To, double>{}
       && !std::is_same<To, rocblas_half>{}
       && !rocblas_is_complex<To> && (alpha_isnan || beta_isnan) && !std::is_same<To, rocblas_f8>{}
       && !std::is_same<To, rocblas_bf8>{})
        return; // Exclude integers or other types which don't support NaN

    float h_alpha_Tc = alpha_isnan ? NAN : arg.get_alpha<float>();
    float h_beta_Tc  = beta_isnan ? NAN : arg.get_beta<float>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    rocblas_local_handle handle{arg};
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row  = transA == rocblas_operation_none ? M : K;
    auto                 A_col  = transA == rocblas_operation_none ? K : M;
    auto                 B_row  = transB == rocblas_operation_none ? K : N;
    auto                 B_col  = transB == rocblas_operation_none ? N : K;
    auto                 d_type = arg.d_type;

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M;

    // NOTE: for mixed of f8 bf8, we will consider it is_bf8 as true since bf8 has less precision than f8
    // TODO: we have not considered f8/b8 with other non-f8 types yet!!
    bool is_bf8
        = arg.composite_compute_type == rocblas_compute_type_bf8_f8_f32
          || arg.composite_compute_type == rocblas_compute_type_f8_bf8_f32
          || arg.composite_compute_type == rocblas_compute_type_bf8_bf8_f32
          || (arg.composite_compute_type == rocblas_compute_type_f32
              && (arg.a_type == rocblas_datatype_bf8_r || arg.b_type == rocblas_datatype_bf8_r));
    bool is_f8
        = arg.composite_compute_type == rocblas_compute_type_f8_f8_f32
          || (arg.composite_compute_type == rocblas_compute_type_f32
              && (arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_f8_r));
    bool is_8bit_float = is_f8 || is_bf8;

    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex3_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  arg.a_type,
                                                  lda,
                                                  nullptr,
                                                  arg.b_type,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  arg.c_type,
                                                  ldc,
                                                  nullptr,
                                                  arg.d_type,
                                                  ldd,
                                                  arg.composite_compute_type,
                                                  algo,
                                                  solution_index,
                                                  flags),
                              rocblas_status_invalid_size);
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_vector<TiA>, dA, (1));
        DEVICE_MEMCHECK(device_vector<TiB>, dB, (1));
        DEVICE_MEMCHECK(device_vector<To>, dC, (1));
        DEVICE_MEMCHECK(device_vector<To>, dD, (1));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex3_fn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &h_alpha_Tc,
                                                dA,
                                                arg.a_type,
                                                lda,
                                                dB,
                                                arg.b_type,
                                                ldb,
                                                &h_beta_Tc,
                                                dC,
                                                arg.c_type,
                                                ldc,
                                                dD,
                                                arg.d_type,
                                                ldd,
                                                arg.composite_compute_type,
                                                algo,
                                                solution_index,
                                                flags));
        return;
    }
#endif

    if(!arg.c_noalias_d)
    {
        ldd    = ldc;
        d_type = arg.c_type;
    }

    // allocate memory on device
    DEVICE_MEMCHECK(device_matrix<TiA>, dA, (A_row, A_col, lda));
    DEVICE_MEMCHECK(device_matrix<TiB>, dB, (B_row, B_col, ldb));
    DEVICE_MEMCHECK(device_matrix<To>, dC, (M, N, ldc));
    device_matrix<To> dD
        = (arg.c_noalias_d) ? device_matrix<To>(M, N, ldd) : device_matrix<To>(0, 1, 1);
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    device_matrix<To>& dDref = (arg.c_noalias_d) ? dD : dC;
    DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc, (1));
    DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc, (1));

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    HOST_MEMCHECK(host_matrix<TiA>, hA, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<TiB>, hB, (B_row, B_col, ldb));
    HOST_MEMCHECK(host_matrix<To>, hC, (M, N, ldc));

    // Initial Data on CPU
    rocblas_seedrand();

    // Initialize data on host memory
    rocblas_init_matrix<TiA>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<TiB>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check || arg.res_check)
    {

        HOST_MEMCHECK(host_matrix<To>, hD_gold, (M, N, ldd));
        HOST_MEMCHECK(host_matrix<To>, hD_1, (M, N, ldd));

        rocblas_init_nan<To>(hD_1, M, N, ldd);
        rocblas_init_nan<To>(hD_gold, M, N, ldd);

        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex3_fn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &h_alpha_Tc,
                                                dA,
                                                arg.a_type,
                                                lda,
                                                dB,
                                                arg.b_type,
                                                ldb,
                                                &h_beta_Tc,
                                                dC,
                                                arg.c_type,
                                                ldc,
                                                dDref,
                                                d_type,
                                                ldd,
                                                arg.composite_compute_type,
                                                algo,
                                                solution_index,
                                                flags));

        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

#if 0
        //if(std::is_same<Ti, rocblas_f8>{} || std::is_same<Ti, rocblas_bf8>{})
        {
            rocblas_cout << "D Matrix1" << std::endl;
            auto* D   = hD_1[0];

            for(int i = 0; i < M; i++)
            {
                for(int j = 0; j < N; j++)
                    rocblas_cout << std::right << std::setw(12) << D[j * ldd + i];
                rocblas_cout << std::endl;
            }
        }
#endif

#if 1
        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        if(!arg.c_noalias_d)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex3_fn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                d_alpha_Tc,
                                                dA,
                                                arg.a_type,
                                                lda,
                                                dB,
                                                arg.b_type,
                                                ldb,
                                                d_beta_Tc,
                                                dC,
                                                arg.c_type,
                                                ldc,
                                                dDref,
                                                d_type,
                                                ldd,
                                                arg.composite_compute_type,
                                                algo,
                                                solution_index,
                                                flags));
#endif

        // CPU BLAS
        // copy C matrix into D matrix
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        // Trusted implementation
        // NOTE: testing new design only for f8 type

        cpu_time_used = get_time_us_no_sync();

#define TEST_PARM                                                                              \
    handle, transA, transB, M, N, K, h_alpha_Tc, dA, hA, lda, dB, hB, ldb, h_beta_Tc, hC, ldc, \
        hD_gold, ldd, stochastic_rounding

        // note ignoring return status of call_trusted_gemm_f8 as reference only
        if((arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_f8_r
            && arg.c_type == arg.d_type
            && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                || arg.c_type == rocblas_datatype_f32_r || arg.c_type == rocblas_datatype_f16_r)
            && arg.composite_compute_type == rocblas_compute_type_f32)
           || arg.composite_compute_type == rocblas_compute_type_f8_f8_f32)
            (void)call_trusted_gemm_f8<TiA, TiB, To, rocblas_f8, rocblas_f8, float>(TEST_PARM);
        else if((arg.a_type == rocblas_datatype_bf8_r && arg.b_type == rocblas_datatype_bf8_r
                 && arg.c_type == arg.d_type
                 && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                     || arg.c_type == rocblas_datatype_f32_r
                     || arg.c_type == rocblas_datatype_f16_r)
                 && arg.composite_compute_type == rocblas_compute_type_f32)
                || arg.composite_compute_type == rocblas_compute_type_bf8_bf8_f32)
            (void)call_trusted_gemm_f8<TiA, TiB, To, rocblas_bf8, rocblas_bf8, float>(TEST_PARM);
        else if((arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_bf8_r
                 && arg.c_type == arg.d_type
                 && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                     || arg.c_type == rocblas_datatype_f32_r
                     || arg.c_type == rocblas_datatype_f16_r)
                 && arg.composite_compute_type == rocblas_compute_type_f32)
                || arg.composite_compute_type == rocblas_compute_type_f8_bf8_f32)
            (void)call_trusted_gemm_f8<TiA, TiB, To, rocblas_f8, rocblas_bf8, float>(TEST_PARM);
        else if((arg.a_type == rocblas_datatype_bf8_r && arg.b_type == rocblas_datatype_f8_r
                 && arg.c_type == arg.d_type
                 && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                     || arg.c_type == rocblas_datatype_f32_r
                     || arg.c_type == rocblas_datatype_f16_r)
                 && arg.composite_compute_type == rocblas_compute_type_f32)
                || arg.composite_compute_type == rocblas_compute_type_bf8_f8_f32)
            (void)call_trusted_gemm_f8<TiA, TiB, To, rocblas_bf8, rocblas_f8, float>(TEST_PARM);
        else
            rocblas_cout << "ERROR Trusted combo not found " << std::endl;

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

#if 0
        {
            rocblas_cout << "D gold Matrix" << std::endl;
            auto* D   = hD_gold[0];

            for(int i = 0; i < M; i++)
            {
                for(int j = 0; j < N; j++)
                    rocblas_cout << std::right << std::setw(12) << D[j * ldd + i];
                rocblas_cout << std::endl;
            }
        }

#endif

        //releasing already used host memory
        hA = host_matrix<TiA>();
        hB = host_matrix<TiB>();
        hC = host_matrix<To>();

        // fetch device mode GPU results
        //CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));

        //Testing starts from here
        // ============ 1. testing result for host pointer mode

        // NOTE: it doesn't make sense to use unit_check with large K since we may overflow easily
        // specially when the To is f8/bf8 (always clipped to max_val)
        // Accumulator type is important to calculate the error bound. However, the result will be downcasted to
        // To type. Therefore, we may need to consider the error bound for that as well.

        if(arg.unit_check)
        {
            // NOTE: f8 can represent integer number from 1.0 to 16.0 and bf8 can represent from 1.0 to 8.0.
            // it may only be safe to match the output exactly when K is very small
            // if(is_8bit_float && K > 240)
            //     rocblas_cout << "******* Applying unit_check with large K is not a good idea "
            //                     "(clipped to max_value or overflow)"
            //                  << std::endl;

            if((std::is_same<Tc, rocblas_half>{} && K > 10000) || (is_8bit_float && K > 16))
            {
                // Note: accumulator type for f8/bf8 is still f32. So, the tol would be 0.
                // However, downcasting the output to f8 or f16 may have rounding errors.... near_check should take care it
                const double tol = K * sum_error_tolerance<Tc>;
                //rocblas_cout << "********** Applying unit->near check, K = " << K << " tol = " << tol << std::endl;
                // rocblas_cout << "********** Applying unit->near check, K = " << K << std::endl;
                near_check_general<To, To>(M, N, ldd, hD_gold, hD_1, tol);
            }
            else
            {
                // rocblas_cout << "********** Applying unit check, K = " << K << std::endl;
                unit_check_general<To, To>(M, N, ldd, hD_gold, hD_1);
            }
        }
        else // can apply both res_check and norm_check if not unit_check
        {
            if(arg.res_check)
            {
                // NOTE: Ti and To can be non-f8 types, but the TcA or TcB still can be f8/bf8 types.
                // We need to consider eps for f8/bf8 since we have lost precision when we downcast the input to f8 (TcA, TcB)
                double eps = is_f8 ? get_epsilon<rocblas_f8>()
                                   : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                // NOTE: Accumulation is in f32 and the eps of f32 is very small when compared with eps of f8. So, we are not
                // considering the reduction errors in tolerance here
                //double tolerance = 100 * sqrt(K);
                //double tolerance = 2 * K;
                double tolerance = 2;
                // rocblas_cout << "********** Applying res_check, K = " << K
                //              << " tol*eps = " << tolerance * eps << std::endl;
                res_check<To, To>(M, N, ldd, hD_gold, hD_1, tolerance * eps);
            }

            if(arg.norm_check)
            {
                // NOTE: NORM calcualtion is based on the To values, so we are considering eps for To type here
                // rocblas_cout << "********** Applying norm check, M = " << M << " N = " << N
                //              << " K = " << K << std::endl;
                auto err1
                    = std::abs(norm_check_general<To>('O', M, N, ldd, (To*)hD_gold, (To*)hD_1));
                double eps = is_f8 ? get_epsilon<rocblas_f8>()
                                   : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                double tolerance
                    = 50; // threshold in lapack style testing, value 50 or 100 in some libraries
                size_t minMN = M < N ? M : N;
#ifdef GOOGLE_TEST
                ASSERT_LE(err1, tolerance * eps * minMN);
#endif
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }

        // fetch device pointer mode result
        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

#if 0
        {
            rocblas_cout << "D Matrix2" << std::endl;

            for(int i = 0; i < M; i++)
            {
                for(int j = 0; j < N; j++)
                    rocblas_cout << std::right << std::setw(12) << hD_1[j * ldd + i];
                rocblas_cout << std::endl;
            }
        }
#endif

        // ============ 2. testing result for device pointer mode

        // NOTE: it doesn't make sense to use unit_check with large K since we may overflow easily
        // specially when the To is f8/bf8 (always clipped to max_val)
        // Accumulator type is important to calculate the error bound. However, the result will be downcasted to
        // To type. Therefore, we may need to consider the error bound for that as well.
        if(arg.unit_check)
        {
            if((std::is_same<Tc, rocblas_half>{} && K > 10000) || (is_8bit_float && K > 16))
            {
                // Note: accumulator type for f8/bf8 is still f32. So, the tol would be 0.
                // However, downcasting the output to f8 or f16 may have rounding errors.... near_check should take care it
                //const double tol = sqrt(K) * sum_error_tolerance<Tc>;
                const double tol = K * sum_error_tolerance<Tc>;
                //rocblas_cout << "********** Applying unit->near check (device_ptr), K = " << K << " tol = " << tol << std::endl;
                near_check_general<To, To>(M, N, ldd, hD_gold, hD_1, tol);
            }
            else
            {
                //rocblas_cout << "********** Applying unit check, K = " << K << std::endl;
                unit_check_general<To, To>(M, N, ldd, hD_gold, hD_1);
            }
        }
        else // not unit_check
        {
            if(arg.res_check)
            {
                // NOTE: Ti and To can be non-f8 types, but the TcA or TcB still can be f8/bf8 types.
                // We need to consider eps for f8/bf8 since we have lost precision when we downcast the input to f8 (TcA, TcB)
                double eps = is_f8 ? get_epsilon<rocblas_f8>()
                                   : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                // NOTE: Accumulation is in f32 and the eps of f32 is very small when compared with eps of f8. So, we are not
                // considering the reduction errors in tolerance here
                //double tolerance = 100 * sqrt(K);
                double tolerance = 2;
                res_check<To, To>(M, N, ldd, hD_gold, hD_1, tolerance * eps);
            }
            if(arg.norm_check)
            {
                // rocblas_cout << "********** Applying norm check, K = " << K << std::endl;
                auto err1
                    = std::abs(norm_check_general<To>('O', M, N, ldd, (To*)hD_gold, (To*)hD_1));
                double eps = is_f8 ? get_epsilon<rocblas_f8>()
                                   : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                double tolerance
                    = 50; // threshold in lapack style testing, value 50 or 100 in some libraries
                size_t minMN = M < N ? M : N; // should consider K?
#ifdef GOOGLE_TEST
                ASSERT_LE(err1, tolerance * eps * minMN);
#endif
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ex3_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    &h_alpha_Tc,
                                                    dA,
                                                    arg.a_type,
                                                    lda,
                                                    dB,
                                                    arg.b_type,
                                                    ldb,
                                                    &h_beta_Tc,
                                                    dC,
                                                    arg.c_type,
                                                    ldc,
                                                    dDref,
                                                    d_type,
                                                    ldd,
                                                    arg.composite_compute_type,
                                                    algo,
                                                    solution_index,
                                                    flags));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_ex3_fn(handle,
                                transA,
                                transB,
                                M,
                                N,
                                K,
                                &h_alpha_Tc,
                                dA,
                                arg.a_type,
                                lda,
                                dB,
                                arg.b_type,
                                ldb,
                                &h_beta_Tc,
                                dC,
                                arg.c_type,
                                ldc,
                                dDref,
                                d_type,
                                ldd,
                                arg.composite_compute_type,
                                algo,
                                solution_index,
                                flags);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldb,
                      e_ldc,
                      e_ldd,
                      e_batch_count>{}
            .log_args<To>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          gemm_gflop_count<Tc>(M, N, K),
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocblas_error);
    }
}

#undef TEST_PARM

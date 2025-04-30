/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "../blas3/rocblas_gemm.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"
#include <random>

#define EX_TYPECASTING_PARM                                                                    \
    handle, trans_a, trans_b, m, n, k, alpha, a, offsetAin, lda, stride_a, b, offsetBin, ldb,  \
        stride_b, beta, c, offsetCin, ldc, stride_c, d, offsetDin, ldd, stride_d, batch_count, \
        rocblas_gemm_flags(flags)

#define EX_TEMPLATE_PARM                                                                           \
    handle, trans_a, trans_b, m, n, k, alpha, a, a_type, offsetAin, lda, stride_a, b, b_type,      \
        offsetBin, ldb, stride_b, beta, c, c_type, offsetCin, ldc, stride_c, d, d_type, offsetDin, \
        ldd, stride_d, batch_count, compute_type, flags

template <bool BATCHED>
rocblas_status rocblas_gemm_ex3_template(rocblas_handle      handle,
                                         rocblas_operation   trans_a,
                                         rocblas_operation   trans_b,
                                         rocblas_int         m,
                                         rocblas_int         n,
                                         rocblas_int         k,
                                         const void*         alpha,
                                         const void*         a,
                                         rocblas_datatype    a_type,
                                         rocblas_int         offsetAin,
                                         rocblas_int         lda,
                                         rocblas_stride      stride_a,
                                         const void*         b,
                                         rocblas_datatype    b_type,
                                         rocblas_int         offsetBin,
                                         rocblas_int         ldb,
                                         rocblas_stride      stride_b,
                                         const void*         beta,
                                         const void*         c,
                                         rocblas_datatype    c_type,
                                         rocblas_int         offsetCin,
                                         rocblas_int         ldc,
                                         rocblas_stride      stride_c,
                                         void*               d,
                                         rocblas_datatype    d_type,
                                         rocblas_int         offsetDin,
                                         rocblas_int         ldd,
                                         rocblas_stride      stride_d,
                                         rocblas_int         batch_count,
                                         rocblas_computetype compute_type,
                                         uint32_t            flags);

/*
 *  Pseudo random number generator
 *      Works only when T is f32 or f16
 */
template <typename T,
          std::enable_if_t<(std::is_same<float, T>{} || std::is_same<rocblas_half, T>{}), int> = 0>
ROCBLAS_KERNEL_ILF uint32_t prand_generator(int64_t tid, uint32_t seed, T val)
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
ROCBLAS_KERNEL_ILF uint32_t prand_generator(int64_t tid, uint32_t seed, T val)
{
    return 0;
}

/***************************************************************************************
    Quantization: New single matrix conversion routine
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
ROCBLAS_KERNEL(DIM_M* DIM_N) general_conversion_kernel(rocblas_int    M,
                                                       rocblas_int    N,
                                                       const TiA*     dA_array,
                                                       TgA*           dA_array_new,
                                                       rocblas_int    lda,
                                                       rocblas_int    lda_new,
                                                       rocblas_stride stride_a,
                                                       rocblas_int    batch_count,
                                                       uint32_t       seedA)
{
    int x = ((TRANS_A == 'N') ? BLK_M : BLK_N) * blockIdx.x + threadIdx.x;
    int y = ((TRANS_A == 'N') ? BLK_N : BLK_M) * blockIdx.y + threadIdx.y;

    int blz = blockIdx.z; // block's matrix in the batch

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);

    // conversion
    for(int m = 0; m < BLK_M; m += DIM_M)
    {
        for(int n = 0; n < BLK_N; n += DIM_N)
        {
            int64_t i = (TRANS_A == 'N') ? (x + m) : (x + n);
            int64_t j = (TRANS_A == 'N') ? (y + n) : (y + m);

            if(i < ((TRANS_A == 'N') ? M : N) && j < ((TRANS_A == 'N') ? N : M))
            {
                if(TRANS_A != 'C')
                {
                    int64_t  gid = i + j * size_t(lda);
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator<TiA>(gid, seedA, dA[i + j * size_t(lda)]);

                    dA_array_new[i + j * lda_new]
                        = TgA(explicit_downcast<TcA, TiA, stochastic_rounding>(
                            dA[i + j * size_t(lda)], rng));
                }
                else if(TRANS_A == 'C')
                {
                    int64_t  gid = i + j * size_t(lda);
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator<TiA>(gid, seedA, conj(dA[i + j * size_t(lda)]));

                    dA_array_new[i + j * size_t(lda_new)]
                        = TgA(explicit_downcast<TcA, TiA, stochastic_rounding>(
                            conj(dA[i + j * size_t(lda)]), rng));
                }
            }
        }
    }
}

/*
 *  Generalized F8 GEMM_EX kernel : HIP_GEMM
 *  NOTE: it is very slow when compare with Tensile's GEMM
 *        We don't expect to call it any time except for the fall-back cases, e.g., M,N,K too large to allocate
 *        workspace for quantization.
 *
 */
template <
    typename TiA,
    typename TiB,
    typename To,
    typename TcA, // computeAtype
    typename TcB, // computeBtype
    typename Tacc, // accumulator type
    int  DIM_M,
    int  DIM_N,
    int  BLK_M,
    int  BLK_N,
    int  BLK_K,
    int  DIM_M_A,
    int  DIM_N_A,
    int  DIM_M_B,
    int  DIM_N_B,
    bool BETA_EQ_ZERO,
    char TRANS_A,
    char TRANS_B,
    bool stochastic_rounding,
    std::enable_if_t<(std::is_same<float, Tacc>{}
                      && (std::is_same<rocblas_f8, TcA>{} || std::is_same<rocblas_bf8, TcA>{})
                      && (std::is_same<rocblas_f8, TcB>{} || std::is_same<rocblas_bf8, TcB>{})),
                     int> = 0>
__attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N)))
ROCBLAS_KERNEL(DIM_M* DIM_N)
    gemm_batched_general_kernel(rocblas_int    M,
                                rocblas_int    N,
                                rocblas_int    K,
                                const Tacc     alpha,
                                const TiA*     dA_array, // NOTE: may work only for non-batch
                                rocblas_int    lda,
                                rocblas_stride stride_a,
                                const TiB*     dB_array,
                                rocblas_int    ldb,
                                rocblas_stride stride_b,
                                const Tacc     beta,
                                const To*      dC_array,
                                rocblas_int    ldc,
                                rocblas_stride stride_c,
                                To*            dD_array,
                                rocblas_int    ldd,
                                rocblas_stride stride_d,
                                rocblas_int    batch_count,
                                uint32_t       seedA,
                                uint32_t       seedB,
                                uint32_t       seedC)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_M_A; // thread's m position for loading A
    int thyA = idt / DIM_M_A; // thread's n position for loading A
    int thxB = idt % DIM_M_B; // thread's m position for loading B
    int thyB = idt / DIM_M_B; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);
    auto* dD = load_ptr_batch(dD_array, blz, 0, stride_d);

    __shared__ TcA sA[BLK_K][BLK_M]; // shared memory for A
    __shared__ TcB sB[BLK_N][BLK_K]; // shared memory for B

    Tacc rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

    int a_i_offset = thxA + BLK_M * blx;
    int a_j_offset = thyA;
    int b_i_offset = thxB;
    int b_j_offset = thyB + BLK_N * bly;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        for(int n = 0; n < BLK_K; n += DIM_N_A)
        {
            for(int m = 0; m < BLK_M; m += DIM_M_A)
            {
                int i = m + a_i_offset;
                int j = n + kk + a_j_offset;
                if(i < M && j < K)
                {
                    // operator overloading from Ti to TcA
                    if(TRANS_A == 'N')
                    {
                        int      gid = i + j * lda;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiA>(gid, seedA, dA[i + j * size_t(lda)]);

                        sA[n + thyA][m + thxA] = explicit_downcast<TcA, TiA, stochastic_rounding>(
                            dA[i + j * size_t(lda)], rng);
                    }
                    else if(TRANS_A == 'T')
                    {
                        int      gid = i * lda + j;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiA>(gid, seedA, dA[i * size_t(lda) + j]);

                        sA[n + thyA][m + thxA] = explicit_downcast<TcA, TiA, stochastic_rounding>(
                            dA[i * size_t(lda) + j], rng);
                    }
                    else if(TRANS_A == 'C')
                    {
                        int      gid = i * lda + j;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiA>(gid, seedA, conj(dA[i * size_t(lda) + j]));

                        sA[n + thyA][m + thxA] = explicit_downcast<TcA, TiA, stochastic_rounding>(
                            conj(dA[i * size_t(lda) + j]), rng);
                    }
                }
                else
                {
                    sA[n + thyA][m + thxA] = TcA(0.0);
                }
            }
        }

        for(int n = 0; n < BLK_N; n += DIM_N_B)
        {
            for(int m = 0; m < BLK_K; m += DIM_M_B)
            {
                int i = m + kk + b_i_offset;
                int j = n + b_j_offset;

                if(i < K && j < N)
                {
                    if(TRANS_B == 'N')
                    {
                        int      gid = i + j * ldb;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiB>(gid, seedB, dB[i + j * size_t(ldb)]);

                        sB[n + thyB][m + thxB] = explicit_downcast<TcB, TiB, stochastic_rounding>(
                            dB[i + j * size_t(ldb)], rng);
                    }
                    else if(TRANS_B == 'T')
                    {
                        int      gid = i * ldb + j;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiB>(gid, seedB, dB[i * size_t(ldb) + j]);

                        sB[n + thyB][m + thxB] = explicit_downcast<TcB, TiB, stochastic_rounding>(
                            dB[i * size_t(ldb) + j], rng);
                    }
                    else if(TRANS_B == 'C')
                    {
                        int      gid = i * ldb + j;
                        uint32_t rng = 0;
                        if(stochastic_rounding)
                            rng = prand_generator<TiB>(gid, seedB, conj(dB[i * size_t(ldb) + j]));

                        sB[n + thyB][m + thxB] = explicit_downcast<TcB, TiB, stochastic_rounding>(
                            conj(dB[i * size_t(ldb) + j]), rng);
                    }
                }
                else
                {
                    sB[n + thyB][m + thxB] = TcB(0.0);
                }
            }
        }

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_M / DIM_M; ++m)
        {
            int coord_dCm = blx * BLK_M + m * DIM_M + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;
            if(coord_dCn < N && coord_dCm < M)
            {
                if(BETA_EQ_ZERO)
                {
                    int      gid = coord_dCn * ldc + coord_dCm;
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator<Tacc>(gid, seedC, alpha * rC[n][m]);

                    dD[coord_dCn * ldc + coord_dCm]
                        = explicit_downcast<To, Tacc, stochastic_rounding>(alpha * rC[n][m], rng);
                }
                else
                {
                    int      gid = coord_dCn * ldc + coord_dCm;
                    uint32_t rng = 0;
                    if(stochastic_rounding)
                        rng = prand_generator<Tacc>(
                            gid,
                            seedC,
                            alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm]);

                    dD[coord_dCn * ldc + coord_dCm]
                        = explicit_downcast<To, Tacc, stochastic_rounding>(
                            (alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm]),
                            rng);
                }
            }
        }
    }
}

/***************************************************************************************/

inline rocblas_status rocblas_internal_f16_conversion__mem(rocblas_handle              handle,
                                                           std::pair<bool, size_t>     a_mem,
                                                           std::pair<bool, size_t>     b_mem,
                                                           std::pair<bool, size_t>     c_mem,
                                                           std::pair<bool, size_t>     gsu_mem,
                                                           rocblas_device_malloc_base& w_mem,
                                                           void*&                      w_mem_TA,
                                                           void*&                      w_mem_TB,
                                                           void*&                      w_mem_TD)
{
    auto& workspace = static_cast<decltype(handle->device_malloc(0))&>(w_mem);

    rocblas_status memory_status = rocblas_status_success;

    // current kernel call is a size query
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(a_mem.first ? a_mem.second : 0,
                                                      b_mem.first ? b_mem.second : 0,
                                                      c_mem.first ? c_mem.second : 0);
    }
    // allocate memory
    workspace = gsu_mem.first ? handle->device_malloc_with_GSU(a_mem.first ? a_mem.second : 0,
                                                               b_mem.first ? b_mem.second : 0,
                                                               c_mem.first ? c_mem.second : 0,
                                                               gsu_mem.second)
                              : handle->device_malloc(a_mem.first ? a_mem.second : 0,
                                                      b_mem.first ? b_mem.second : 0,
                                                      c_mem.first ? c_mem.second : 0);

    if(!workspace)
        return rocblas_status_memory_error;
    if(a_mem.first)
        w_mem_TA = workspace[0];
    if(b_mem.first)
        w_mem_TB = workspace[1];
    if(c_mem.first)
        w_mem_TD = workspace[2];

    return memory_status;
}

template <bool BATCHED,
          typename TiA,
          typename TiB = TiA,
          typename To  = TiA,
          typename TcA,
          typename TcB,
          typename Tacc>
rocblas_status gemm_ex3_fallback(rocblas_handle     handle,
                                 rocblas_operation  trans_a,
                                 rocblas_operation  trans_b,
                                 rocblas_int        m,
                                 rocblas_int        n,
                                 rocblas_int        k,
                                 const void*        alpha,
                                 const void*        a,
                                 rocblas_int        offsetAin,
                                 rocblas_int        lda,
                                 rocblas_stride     stride_a,
                                 const void*        b,
                                 rocblas_int        offsetBin,
                                 rocblas_int        ldb,
                                 rocblas_stride     stride_b,
                                 const void*        beta,
                                 const void*        c,
                                 rocblas_int        offsetCin,
                                 rocblas_int        ldc,
                                 rocblas_stride     stride_c,
                                 void*              d,
                                 rocblas_int        offsetDin,
                                 rocblas_int        ldd,
                                 rocblas_stride     stride_d,
                                 rocblas_int        batch_count,
                                 rocblas_gemm_flags flags)
{
    float alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
        return rocblas_status_invalid_size;

    bool     stochastic_rounding = flags & rocblas_gemm_flags_stochastic_rounding;
    uint32_t seedA = 0, seedB = 0, seedC = 0;
    if(stochastic_rounding)
    {
        std::random_device                      rd;
        std::mt19937                            gen(rd());
        std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);

        const char* sr_seed_string_a = std::getenv("SR_SEED_A");
        const char* sr_seed_string_b = std::getenv("SR_SEED_B");
        const char* sr_seed_string_c = std::getenv("SR_SEED_C");

        if(sr_seed_string_a != NULL)
            seedA = std::strtol(sr_seed_string_a, NULL, 10);
        else
            seedA = distribution(gen);

        if(sr_seed_string_b != NULL)
            seedB = std::strtol(sr_seed_string_b, NULL, 10);
        else
            seedB = distribution(gen);

        if(sr_seed_string_c != NULL)
            seedC = std::strtol(sr_seed_string_c, NULL, 10);
        else
            seedC = distribution(gen);
    }

    hipStream_t stream = handle->get_stream();
    const int   dim_m  = 16;
    const int   dim_n  = 16;
    const int   blk_m  = 32;
    const int   blk_n  = 32;
    const int   blk_k  = 8;
    dim3        dimBlock(dim_m, dim_n, 1);
    dim3        dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count);

    if((*((Tacc*)beta)) == 0) // check the deref value of beta, not the ptr
    {
        // clang-format off
    if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                <TiA,
                                TiB,
                                To,
                                TcA,
                                TcB,
                                float,
                                dim_m,
                                dim_n,
                                blk_m,
                                blk_n,
                                blk_k,
                                blk_m,
                                blk_k,
                                blk_k,
                                blk_n,
                                true,
                                'N',
                                'N',
                                true>),
            dimGrid,
            dimBlock,
            0,
            stream,
            m,
            n,
            k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda,
            stride_a,
            (const TiB *) b,
            ldb,
            stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc,
            stride_c,
            (To *) d,
            ldd,
            stride_d,
            batch_count,
            seedA, seedB, seedC);

        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                <TiA,
                                TiB,
                                To,
                                TcA,
                                TcB,
                                float,
                                dim_m,
                                dim_n,
                                blk_m,
                                blk_n,
                                blk_k,
                                blk_m,
                                blk_k,
                                blk_k,
                                blk_n,
                                true,
                                'N',
                                'N',
                                false>),
            dimGrid,
            dimBlock,
            0,
            stream,
            m,
            n,
            k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda,
            stride_a,
            (const TiB *) b,
            ldb,
            stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc,
            stride_c,
            (To *) d,
            ldd,
            stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'N', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);

        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'N', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);


    }
    else if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true,'N', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true,'N', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);

    }

    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'N', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'N', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // no SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }
        // clang-format on
    }
    else // beta non zero
    {
        // clang-format off
    if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
                    dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'N', true>),
                    dimGrid,
                    dimBlock,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    *((const Tacc *) alpha),
                    (const TiA *) a,
                    lda, stride_a,
                    (const TiB *) b,
                    ldb,
                    stride_b,
                    *((const Tacc *) beta),
                    (const To *) c,
                    ldc,
                    stride_c,
                    (To *) d,
                    ldd,
                    stride_d,
                    batch_count,
                    seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
                    dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'N', false>),
                    dimGrid,
                    dimBlock,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    *((const Tacc *) alpha),
                    (const TiA *) a,
                    lda, stride_a,
                    (const TiB *) b,
                    ldb,
                    stride_b,
                    *((const Tacc *) beta),
                    (const To *) c,
                    ldc,
                    stride_c,
                    (To *) d,
                    ldd,
                    stride_d,
                    batch_count,
                    seedA, seedB, seedC);
    }
    else if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'N', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'N', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }

    else if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);

    }
    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }
    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'N', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'N', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }
    else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'T', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'T', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
    }
    else if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);

    }
    else if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
    {
        if (stochastic_rounding)
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'C', false>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);
        else // non SR
            ROCBLAS_LAUNCH_KERNEL((gemm_batched_general_kernel
                                    <TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    float,
            dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'C', true>),
            dimGrid, dimBlock, 0, stream, m, n, k,
            *((const Tacc *) alpha),
            (const TiA *) a,
            lda, stride_a,
            (const TiB *) b,
            ldb, stride_b,
            *((const Tacc *) beta),
            (const To *) c,
            ldc, stride_c,
            (To *) d,
            ldd, stride_d,
            batch_count,
            seedA, seedB, seedC);

    }
        // clang-format on
    }

    return rocblas_status_success;
}

template <bool BATCHED,
          typename TiA,
          typename TiB = TiA,
          typename To  = TiA,
          typename TcA,
          typename TcB,
          typename Tacc,
          typename To_expected = Tacc> // To_expected is type expected to return from Tensile kernel
rocblas_status gemm_ex3_quantize(rocblas_handle     handle,
                                 rocblas_operation  trans_a,
                                 rocblas_operation  trans_b,
                                 rocblas_int        m,
                                 rocblas_int        n,
                                 rocblas_int        k,
                                 const void*        alpha,
                                 const void*        a,
                                 rocblas_int        offsetAin,
                                 rocblas_int        lda,
                                 rocblas_stride     stride_a,
                                 const void*        b,
                                 rocblas_int        offsetBin,
                                 rocblas_int        ldb,
                                 rocblas_stride     stride_b,
                                 const void*        beta,
                                 const void*        c,
                                 rocblas_int        offsetCin,
                                 rocblas_int        ldc,
                                 rocblas_stride     stride_c,
                                 void*              d,
                                 rocblas_int        offsetDin,
                                 rocblas_int        ldd,
                                 rocblas_stride     stride_d,
                                 rocblas_int        batch_count,
                                 rocblas_gemm_flags flags)
{
    float alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
        return rocblas_status_invalid_size;

    bool fallback = (trans_a == rocblas_operation_transpose
                     && trans_b == rocblas_operation_transpose && n < 4)
                    || (trans_a == rocblas_operation_none
                        && (m < 4 || (trans_b == rocblas_operation_transpose && n < 4)));

    if(fallback)
        return gemm_ex3_fallback<BATCHED, TiA, TiB, To, TcA, TcB, Tacc>(EX_TYPECASTING_PARM);

    bool stochastic_rounding = flags & rocblas_gemm_flags_stochastic_rounding;

    rocblas_int lda_new = trans_a == rocblas_operation_none ? m : k;
    rocblas_int ldb_new = trans_b == rocblas_operation_none ? k : n;
    rocblas_int ldd_new = m;

    //create new memory

    auto   w_mem        = handle->device_malloc(0);
    void*  w_mem_dTA    = nullptr;
    void*  w_mem_dTB    = nullptr;
    void*  w_mem_dTD    = nullptr;
    bool   GSU_request  = false;
    size_t memsize      = 0;
    bool   To_is_final  = std::is_same<To_expected, To>{};
    bool   TiA_is_final = std::is_same<TiA, TcA>{};
    bool   TiB_is_final = std::is_same<TiB, TcB>{};

    {
        // finding space for GSU
        int32_t           solution_index = 0;
        uint32_t          flag           = 0;
        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;

        rocblas_start_device_memory_size_query(handle);

        rocblas_stride stride_a{1}, stride_b{1}, stride_c{1}, stride_d{1};
        rocblas_gemm_ex3_template<false>(handle,
                                         trans_a,
                                         trans_b,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         w_mem_dTA,
                                         rocblas_datatype_from_type<TcA>,
                                         0,
                                         lda_new,
                                         stride_a,
                                         w_mem_dTB,
                                         rocblas_datatype_from_type<TcB>,
                                         0,
                                         ldb_new,
                                         stride_b,
                                         beta,
                                         w_mem_dTD,
                                         rocblas_datatype_from_type<To_expected>,
                                         0,
                                         ldd_new,
                                         stride_c,
                                         w_mem_dTD,
                                         rocblas_datatype_from_type<To_expected>,
                                         0,
                                         ldd_new,
                                         stride_d,
                                         batch_count,
                                         rocblas_compute_type_f32,
                                         flags);

        rocblas_stop_device_memory_size_query(handle, &memsize);
        if(memsize)
            GSU_request = true;
    }

    size_t size_a, size_b, size_c;

    size_a            = size_t(m) * k;
    size_b            = size_t(k) * n;
    size_c            = size_t(m) * n;
    float local_beta  = *(const float*)beta;
    float local_alpha = *(const float*)alpha;

    rocblas_status status = rocblas_internal_f16_conversion__mem(
        handle,
        std::pair<bool, size_t>(!TiA_is_final, size_a * sizeof(TcA)),
        std::pair<bool, size_t>(!TiB_is_final, size_b * sizeof(TcB)),
        std::pair<bool, size_t>(!To_is_final, size_c * sizeof(To_expected)),
        std::pair<bool, size_t>(GSU_request, memsize),
        w_mem,
        w_mem_dTA,
        w_mem_dTB,
        w_mem_dTD);

    if(status != rocblas_status_success)
        return status;

    //call conversion kernel
    hipStream_t stream = handle->get_stream();
    const int   dim_m  = 16;
    const int   dim_n  = 16;
    const int   blk_m  = 32;
    const int   blk_n  = 32;
    const int   blk_k  = 32;

    uint32_t seedA = 0, seedB = 0, seedC = 0;
    if(stochastic_rounding)
    {
        std::random_device                      rd;
        std::mt19937                            gen(rd());
        std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);

        const char* sr_seed_string_a = std::getenv("SR_SEED_A");
        const char* sr_seed_string_b = std::getenv("SR_SEED_B");
        const char* sr_seed_string_c = std::getenv("SR_SEED_C");

        if(sr_seed_string_a != NULL)
            seedA = std::strtol(sr_seed_string_a, NULL, 10);
        else
            seedA = distribution(gen);

        if(sr_seed_string_b != NULL)
            seedB = std::strtol(sr_seed_string_b, NULL, 10);
        else
            seedB = distribution(gen);

        if(sr_seed_string_c != NULL)
            seedC = std::strtol(sr_seed_string_c, NULL, 10);
        else
            seedC = distribution(gen);
    }

    // A conversion
    // clang-format off
    if(!TiA_is_final && local_alpha)
    {
        dim3 dimBlock(dim_m, dim_n, 1);
        int m_block = ((m - 1) / blk_m) + 1;
        int k_block = ((k - 1) / blk_k) + 1;
        dim3 dimGrid(rocblas_operation_none == trans_a ? m_block:k_block, rocblas_operation_none == trans_a ? k_block:m_block, batch_count);

        if(rocblas_operation_none == trans_a)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
        }
        else if(rocblas_operation_transpose == trans_a)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
        }
        else if(rocblas_operation_conjugate_transpose == trans_a)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiA,
                                                              TcA,
                                                              TcA,    // Tensile gemm's TiA
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
                                                              m,
                                                              k,
                                                              (const TiA*)a,
                                                              (TcA*)w_mem_dTA,
                                                              lda,
                                                              lda_new,
                                                              stride_a,
                                                              batch_count,
                                                              seedA);
        }
    }

    // *** B conversion
    if(!TiB_is_final && local_alpha)
    {
        dim3 dimBlock(dim_m, dim_n, 1);
        int k_block = ((k - 1) / blk_k) + 1;
        int n_block = ((n - 1) / blk_n) + 1;
        dim3 dimGrid(rocblas_operation_none == trans_b ? k_block:n_block, rocblas_operation_none == trans_b ? n_block:k_block, batch_count);

        if(rocblas_operation_none == trans_b)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
        }
        else if(rocblas_operation_transpose == trans_b)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
        }
        else if(rocblas_operation_conjugate_transpose == trans_b)
        {
            if(stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
            else if(!stochastic_rounding)
                ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<TiB,
                                                              TcB,
                                                              TcB,    // Tensile gemm's TiB
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
                                                              k,
                                                              n,
                                                              (const TiB*)b,
                                                              (TcB*)w_mem_dTB,
                                                              ldb,
                                                              ldb_new,
                                                              stride_b,
                                                              batch_count,
                                                              seedB);
        }
    }
    // clang-format on

    // C conversion
    // clang-format off
    if(!To_is_final && local_beta!=0)
    {
        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count);

        ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<To,
                                                        To,
                                                        To_expected,    // Tensile gemm's To
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
                                                        m,
                                                        n,
                                                        (const To*)c,
                                                        (To_expected*)w_mem_dTD,
                                                        ldc,
                                                        ldd_new,
                                                        stride_c,
                                                        batch_count,
                                                        seedC);

    }

    //call gemm

    int32_t           solution_index = 0;
    rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;

    status = rocblas_gemm_ex3_template<false>(handle,
                                           trans_a,
                                           trans_b,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           TiA_is_final ? a : w_mem_dTA,
                                           rocblas_datatype_from_type<TcA>,
                                           0,
                                           TiA_is_final ? lda : lda_new,
                                           stride_a,
                                           TiB_is_final ? b : w_mem_dTB,
                                           rocblas_datatype_from_type<TcB>,
                                           0,
                                           TiB_is_final ? ldb :ldb_new,
                                           stride_b,
                                           beta,
                                           To_is_final ? c : w_mem_dTD, //trying for beta = 0
                                           rocblas_datatype_from_type<To_expected>, //rocblas_datatype_f16_r
                                           0,
                                           To_is_final ? ldc : ldd_new,
                                           stride_c,
                                           To_is_final ? d : w_mem_dTD,
                                           rocblas_datatype_from_type<To_expected>, //rocblas_datatype_f16_r
                                           0,
                                           To_is_final ? ldd : ldd_new,
                                           stride_d,
                                           batch_count,
                                           rocblas_compute_type_f32,
                                           flags);

    if(!To_is_final)
    {
        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count);


        ROCBLAS_LAUNCH_KERNEL((general_conversion_kernel<To_expected,
                                                        To,
                                                        To,    // final output
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
                                                        m,
                                                        n,
                                                        (const To_expected*)w_mem_dTD, //rocblas_half
                                                        (To*)d,
                                                        ldd_new,
                                                        ldd,
                                                        stride_d,
                                                        batch_count,
                                                        seedA);
    }

        return status;
}

template <typename TiA,
          typename TiB = TiA,
          typename To  = TiA,
          typename TcA,
          typename TcB,
          typename Tacc>
rocblas_status gemm_ex3_tensile(rocblas_handle     handle,
                                         rocblas_operation  trans_a,
                                         rocblas_operation  trans_b,
                                         rocblas_int        m,
                                         rocblas_int        n,
                                         rocblas_int        k,
                                         const Tacc*        alpha,
                                         const TiA*         a,
                                         rocblas_int        offset_a,
                                         rocblas_int        lda,
                                         rocblas_stride     stride_a,
                                         const TiB*         b,
                                         rocblas_int        offset_b,
                                         rocblas_int        ldb,
                                         rocblas_stride     stride_b,
                                         const Tacc*        beta,
                                         const To*          c,
                                         rocblas_int        offset_c,
                                         rocblas_int        ldc,
                                         rocblas_stride     stride_c,
                                         To*                d,
                                         rocblas_int        offset_d,
                                         rocblas_int        ldd,
                                         rocblas_stride     stride_d,
                                         rocblas_int        batch_count,
                                         rocblas_gemm_flags flags)
{
    RocblasContractionProblem<TiA, To, Tacc, TiB, TcA, TcB> problem{
        handle,   trans_a, trans_b,  m,        n,           k,        alpha,    a,
        nullptr,  lda,     stride_a, offset_a, b,           nullptr,  ldb,      stride_b,
        offset_b, beta,    c,        nullptr,  ldc,         stride_c, offset_c, d,
        nullptr,  ldd,     stride_d, offset_d, batch_count, true,     flags};

    return runContractionProblem(problem);
}

template <bool BATCHED,
          typename TiA,
          typename TiB = TiA,
          typename To  = TiA,
          typename TcA,
          typename TcB,
          typename Tacc>
rocblas_status gemm_ex3_typecasting_tensile(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const void*       alpha,
                                           const void*       a,
                                           rocblas_int       offsetAin,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           const void*       b,
                                           rocblas_int       offsetBin,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           const void*       beta,
                                           const void*       c,
                                           rocblas_int       offsetCin,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           void*             d,
                                           rocblas_int       offsetDin,
                                           rocblas_int       ldd,
                                           rocblas_stride    stride_d,
                                           rocblas_int       batch_count,
                                           rocblas_gemm_flags flags)
{
    Tacc alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;

    if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
        return rocblas_status_invalid_size;

    bool fallback = (trans_a == rocblas_operation_transpose && trans_b == rocblas_operation_transpose && n<4) ||
            (trans_a == rocblas_operation_none && (m<4 || (trans_b == rocblas_operation_transpose && n<4)));

    if(check_numerics && !std::is_same_v<TiA, signed char> && !std::is_same_v<TiB, signed char>)
    {
        bool           is_input = true;
        rocblas_status gemm_ex_check_numerics_status
            = rocblas_gemm_check_numerics(BATCHED ? "rocblas_gemm_batched_ex3" : (batch_count > 1 ? "rocblas_gemm_strided_batched_ex3" : "rocblas_gemm_ex3"),
                                            handle,
                                            trans_a,
                                            trans_b,
                                            m,
                                            n,
                                            k,
                                            (const TiA*)a,
                                            offsetAin,
                                            lda,
                                            stride_a,
                                            (const TiB*)b,
                                            offsetBin,
                                            ldb,
                                            stride_b,
                                            (const To*)c,
                                            offsetCin,
                                            ldc,
                                            stride_c,
                                            batch_count,
                                            check_numerics,
                                            is_input);
        if(gemm_ex_check_numerics_status != rocblas_status_success)
            return gemm_ex_check_numerics_status;
    }

    if(fallback)
        return gemm_ex3_fallback<BATCHED,
                                    TiA,
                                    TiB,
                                    To,
                                    TcA,
                                    TcB,
                                    Tacc>(EX_TYPECASTING_PARM);

    status = gemm_ex3_tensile<TiA, TiB, To, TcA, TcB, Tacc>(handle,
                                                                   trans_a,
                                                                   trans_b,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   (const Tacc*)alpha,
                                                                   (const TiA*)a,
                                                                   offsetAin,
                                                                   lda,
                                                                   stride_a,
                                                                   (const TiB*)b,
                                                                   offsetBin,
                                                                   ldb,
                                                                   stride_b,
                                                                   (const Tacc*)beta,
                                                                   (const To*)c,
                                                                   offsetCin,
                                                                   ldc,
                                                                   stride_c,
                                                                   (To*)d,
                                                                   offsetDin,
                                                                   ldd,
                                                                   stride_d,
                                                                   batch_count,
                                                                   flags);

    if(check_numerics && !std::is_same_v<TiA, signed char> && !std::is_same_v<TiB, signed char>)
    {
        bool           is_input = false;
        rocblas_status gemm_ex_check_numerics_status
            = rocblas_gemm_check_numerics(BATCHED ? "rocblas_gemm_batched_ex3" : (batch_count > 1 ? "rocblas_gemm_strided_batched_ex3" : "rocblas_gemm_ex3"),
                                            handle,
                                            trans_a,
                                            trans_b,
                                            m,
                                            n,
                                            k,
                                            (const TiA*)a,
                                            offsetAin,
                                            lda,
                                            stride_a,
                                            (const TiB*)b,
                                            offsetBin,
                                            ldb,
                                            stride_b,
                                            (const To*)d,
                                            offsetDin,
                                            ldd,
                                            stride_d,
                                            batch_count,
                                            check_numerics,
                                            is_input);
        if(gemm_ex_check_numerics_status != rocblas_status_success)
            return gemm_ex_check_numerics_status;
    }

    return status;
}

template <typename T>
inline rocblas_status validateArgs(rocblas_handle      handle,
                                   rocblas_operation   trans_a,
                                   rocblas_operation   trans_b,
                                   rocblas_int         m,
                                   rocblas_int         n,
                                   rocblas_int         k,
                                   const T*            alpha,
                                   const void*         a,
                                   rocblas_int         ld_a,
                                   const void*         b,
                                   rocblas_int         ld_b,
                                   const T*            beta,
                                   const void*         c,
                                   rocblas_datatype    c_type,
                                   rocblas_int         ld_c,
                                   const void*         d,
                                   rocblas_datatype    d_type,
                                   rocblas_int         ld_d,
                                   rocblas_computetype compute_type,
                                   rocblas_int         batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose
       && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose
       && trans_b != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ld_c < m || ld_d < m || ld_a < (trans_a == rocblas_operation_none ? m : k)
       || ld_b < (trans_b == rocblas_operation_none ? k : n))
        return rocblas_status_invalid_size;

    // quick return
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(handle->is_device_memory_size_query())
        return rocblas_status_continue;

    // pointers must be valid
    if((k && !alpha) || !beta || !d)
        return rocblas_status_invalid_pointer;

    // If C is nullptr, beta must be zero
    if(!c)
    {
        if(*(const float*)beta)
            return rocblas_status_invalid_pointer;
    }

    // If k != 0 and either A or B is nullptr, alpha must be zero
    if(k && (!a || !b))
    {
        if(*(const float*)alpha)
            return rocblas_status_invalid_pointer;
    }

    if(c == d)
    {
        if(ld_c != ld_d)
            return rocblas_status_invalid_size;
        if(c_type != d_type)
            return rocblas_status_invalid_value;
    }

    return rocblas_status_continue;
}

template <bool BATCHED>
rocblas_status rocblas_gemm_ex3_template(rocblas_handle      handle,
                                         rocblas_operation   trans_a,
                                         rocblas_operation   trans_b,
                                         rocblas_int         m,
                                         rocblas_int         n,
                                         rocblas_int         k,
                                         const void*         alpha,
                                         const void*         a,
                                         rocblas_datatype    a_type,
                                         rocblas_int         offsetAin,
                                         rocblas_int         lda,
                                         rocblas_stride      stride_a,
                                         const void*         b,
                                         rocblas_datatype    b_type,
                                         rocblas_int         offsetBin,
                                         rocblas_int         ldb,
                                         rocblas_stride      stride_b,
                                         const void*         beta,
                                         const void*         c,
                                         rocblas_datatype    c_type,
                                         rocblas_int         offsetCin,
                                         rocblas_int         ldc,
                                         rocblas_stride      stride_c,
                                         void*               d,
                                         rocblas_datatype    d_type,
                                         rocblas_int         offsetDin,
                                         rocblas_int         ldd,
                                         rocblas_stride      stride_d,
                                         rocblas_int         batch_count,
                                         rocblas_computetype compute_type,
                                         uint32_t            flags)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(BATCHED)
    {
        stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        stride_c = rocblas_stride(ldc) * n;
        stride_d = rocblas_stride(ldd) * n;
    }

    rocblas_status rb_status = rocblas_status_not_implemented;

    if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r
       && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
       && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_f8,
                                                float,
                                                rocblas_f8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_f8,
                                                rocblas_half,
                                                rocblas_f8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                float,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_half,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                float,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                rocblas_half,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                float,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                rocblas_half,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r
            && c_type == rocblas_datatype_f8_r && d_type == rocblas_datatype_f8_r
            && compute_type == rocblas_compute_type_f32)
    {
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_f8,
                                                rocblas_f8,
                                                rocblas_f8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r
            && compute_type == rocblas_compute_type_f32)
    {
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    }
    // else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r
    //         && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r
    //         && compute_type == rocblas_compute_type_f32)
    // {
    //     rb_status = gemm_ex3_fallback<BATCHED,
    //                                      rocblas_f8,
    //                                      rocblas_f8,
    //                                      rocblas_bf8,
    //                                      rocblas_f8,
    //                                      rocblas_f8,
    //                                      float>(EX_TYPECASTING_PARM);
    // }
    // else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r
    //         && c_type == rocblas_datatype_f8_r && d_type == rocblas_datatype_f8_r
    //         && compute_type == rocblas_compute_type_f32)
    // {
    //     rb_status = gemm_ex3_fallback<BATCHED,
    //                                      rocblas_bf8,
    //                                      rocblas_bf8,
    //                                      rocblas_f8,
    //                                      rocblas_bf8,
    //                                      rocblas_bf8,
    //                                      float>(EX_TYPECASTING_PARM);
    // }
    else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r
            && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                float>(EX_TYPECASTING_PARM);
    else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r
            && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r
            && compute_type == rocblas_compute_type_f32)
        rb_status = gemm_ex3_typecasting_tensile<BATCHED,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                rocblas_bf8,
                                                rocblas_bf8,
                                                rocblas_f8,
                                                float>(EX_TYPECASTING_PARM);
    else if(compute_type == rocblas_compute_type_f8_f8_f32)
    {
        // if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
        //    && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        //     rb_status
        //         = gemm_ex3_fallback<BATCHED, float, float, float, rocblas_f8, rocblas_f8, float>(
        //             EX_TYPECASTING_PARM);
        if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_f8,
                                         rocblas_f8,
                                         float,
                                         rocblas_half>(EX_TYPECASTING_PARM);
        // else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
        //         && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_f8,
        //                                      rocblas_f8,
        //                                      float>(EX_TYPECASTING_PARM);
        else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f32_r
                && c_type == rocblas_datatype_f8_r && d_type == rocblas_datatype_f8_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_f8,
                                         float,
                                         rocblas_f8,
                                         rocblas_f8,
                                         rocblas_f8,
                                         float,
                                         rocblas_f8>(EX_TYPECASTING_PARM);
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f8_r
                && c_type == rocblas_datatype_f8_r && d_type == rocblas_datatype_f8_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         float,
                                         rocblas_f8,
                                         rocblas_f8,
                                         rocblas_f8,
                                         rocblas_f8,
                                         float,
                                         rocblas_f8>(EX_TYPECASTING_PARM);
    }
    else if(compute_type == rocblas_compute_type_f8_bf8_f32)
    {
        // if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
        //    && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      float,
        //                                      float,
        //                                      float,
        //                                      rocblas_f8,
        //                                      rocblas_bf8,
        //                                      float>(EX_TYPECASTING_PARM);
        if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_f8,
                                         rocblas_bf8,
                                         float,
                                         rocblas_half>(EX_TYPECASTING_PARM);
        // else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
        //         && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_f8,
        //                                      rocblas_bf8,
        //                                      float>(EX_TYPECASTING_PARM);
        if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_bf8_r
                && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         float,
                                         rocblas_bf8,
                                         rocblas_bf8,
                                         rocblas_f8,
                                         rocblas_bf8,
                                         float,
                                         rocblas_bf8>(EX_TYPECASTING_PARM);
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_bf8_r
                && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         float,
                                         rocblas_bf8,
                                         float,
                                         rocblas_f8,
                                         rocblas_bf8,
                                         float>(EX_TYPECASTING_PARM);
    }
    else if(compute_type == rocblas_compute_type_bf8_f8_f32)
    {
        // if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
        //    && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      float,
        //                                      float,
        //                                      float,
        //                                      rocblas_bf8,
        //                                      rocblas_f8,
        //                                      float>(EX_TYPECASTING_PARM);
        if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_bf8,
                                         rocblas_f8,
                                         float,
                                         rocblas_half>(EX_TYPECASTING_PARM);
        // else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
        //         && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bf8,
        //                                      rocblas_f8,
        //                                      float>(EX_TYPECASTING_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f32_r
                && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_bf8,
                                         float,
                                         float,
                                         rocblas_bf8,
                                         rocblas_f8,
                                         float>(EX_TYPECASTING_PARM);
    }
    else if(compute_type == rocblas_compute_type_bf8_bf8_f32)
    {
        // if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
        //    && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      float,
        //                                      float,
        //                                      float,
        //                                      rocblas_bf8,
        //                                      rocblas_bf8,
        //                                      float>(EX_TYPECASTING_PARM);
        if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_half,
                                         rocblas_bf8,
                                         rocblas_bf8,
                                         float,
                                         rocblas_half>(EX_TYPECASTING_PARM);
        // else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
        //         && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        //     rb_status = gemm_ex3_fallback<BATCHED,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bfloat16,
        //                                      rocblas_bf8,
        //                                      rocblas_bf8,
        //                                      float>(EX_TYPECASTING_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f32_r
                && c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r)
            rb_status = gemm_ex3_quantize<BATCHED,
                                         rocblas_bf8,
                                         float,
                                         rocblas_bf8,
                                         rocblas_bf8,
                                         rocblas_bf8,
                                         float,
                                         rocblas_bf8>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

#undef EX_TYPECASTING_PARM

template <bool BATCHED, typename TiA,  typename TiB, typename To>
rocblas_status rocblas_gemm_batched_ex3_typecasting(rocblas_handle      handle,
                                         rocblas_operation   trans_a,
                                         rocblas_operation   trans_b,
                                         rocblas_int         m,
                                         rocblas_int         n,
                                         rocblas_int         k,
                                         const void*         alpha,
                                         const void*         a,
                                         rocblas_datatype    a_type,
                                         rocblas_int         offsetAin,
                                         rocblas_int         lda,
                                         rocblas_stride      stride_a,
                                         const void*         b,
                                         rocblas_datatype    b_type,
                                         rocblas_int         offsetBin,
                                         rocblas_int         ldb,
                                         rocblas_stride      stride_b,
                                         const void*         beta,
                                         const void*         c,
                                         rocblas_datatype    c_type,
                                         rocblas_int         offsetCin,
                                         rocblas_int         ldc,
                                         rocblas_stride      stride_c,
                                         void*               d,
                                         rocblas_datatype    d_type,
                                         rocblas_int         offsetDin,
                                         rocblas_int         ldd,
                                         rocblas_stride      stride_d,
                                         rocblas_int         batch_count,
                                         rocblas_computetype compute_type,
                                         uint32_t            flags)
{
    rocblas_status rb_status = rocblas_status_not_implemented;

    if(BATCHED)
    {
        std::unique_ptr<TiA*[]> a_host = std::make_unique<TiA*[]>(batch_count);
        std::unique_ptr<TiB*[]> b_host = std::make_unique<TiB*[]>(batch_count);
        std::unique_ptr<To*[]> c_host = std::make_unique<To*[]>(batch_count);
        std::unique_ptr<To*[]> d_host = std::make_unique<To*[]>(batch_count);

        if(a!=nullptr)
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&a_host[0],
                            a,
                            sizeof(TiA*) * batch_count,
                            hipMemcpyDeviceToHost,
                            handle->get_stream()));
        if(b!=nullptr)
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&b_host[0],
                            b,
                            sizeof(TiB*) * batch_count,
                            hipMemcpyDeviceToHost,
                            handle->get_stream()));
        if(c!=nullptr)
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&c_host[0],
                            c,
                            sizeof(To*) * batch_count,
                            hipMemcpyDeviceToHost,
                            handle->get_stream()));
        if(d!=nullptr)
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&d_host[0],
                            d,
                            sizeof(To*) * batch_count,
                            hipMemcpyDeviceToHost,
                            handle->get_stream()));

        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));

        for(int i = 0; i<batch_count; i++)
        {
            rb_status = rocblas_gemm_ex3_template<true>(handle,
                                                        trans_a,
                                                        trans_b,
                                                        m,
                                                        n,
                                                        k,
                                                        alpha,
                                                        a==nullptr ? a : a_host[i],
                                                        a_type,
                                                        offsetAin,
                                                        lda,
                                                        stride_a,
                                                        b==nullptr ? b :  b_host[i],
                                                        b_type,
                                                        offsetBin,
                                                        ldb,
                                                        stride_b,
                                                        beta,
                                                        c==nullptr ? c :  c_host[i],
                                                        c_type,
                                                        offsetCin,
                                                        ldc,
                                                        stride_c,
                                                        d==nullptr ? d :  d_host[i],
                                                        d_type,
                                                        offsetDin,
                                                        ldd,
                                                        stride_d,
                                                        1,
                                                        compute_type,
                                                        flags);
            if(rb_status!=rocblas_status_success)
                return rb_status;
        }
    }
    else
    {
        for(rocblas_int i = 0; i<batch_count; i++)
        {
            rb_status = rocblas_gemm_ex3_template<false>(handle,
                                                        trans_a,
                                                        trans_b,
                                                        m,
                                                        n,
                                                        k,
                                                        alpha,
                                                        (const TiA *)a + i*stride_a,
                                                        a_type,
                                                        offsetAin,
                                                        lda,
                                                        stride_a,
                                                        (const TiB *)b + i*stride_b,
                                                        b_type,
                                                        offsetBin,
                                                        ldb,
                                                        stride_b,
                                                        beta,
                                                        (const To *)c + i*stride_c,
                                                        c_type,
                                                        offsetCin,
                                                        ldc,
                                                        stride_c,
                                                        (To *)d + i*stride_d,
                                                        d_type,
                                                        offsetDin,
                                                        ldd,
                                                        stride_d,
                                                        1,
                                                        compute_type,
                                                        flags);
            if(rb_status!=rocblas_status_success)
                return rb_status;
        }
    }

    return rb_status;

}

template <bool BATCHED>
rocblas_status rocblas_gemm_batched_ex3_template(rocblas_handle      handle,
                                         rocblas_operation   trans_a,
                                         rocblas_operation   trans_b,
                                         rocblas_int         m,
                                         rocblas_int         n,
                                         rocblas_int         k,
                                         const void*         alpha,
                                         const void*         a,
                                         rocblas_datatype    a_type,
                                         rocblas_int         offsetAin,
                                         rocblas_int         lda,
                                         rocblas_stride      stride_a,
                                         const void*         b,
                                         rocblas_datatype    b_type,
                                         rocblas_int         offsetBin,
                                         rocblas_int         ldb,
                                         rocblas_stride      stride_b,
                                         const void*         beta,
                                         const void*         c,
                                         rocblas_datatype    c_type,
                                         rocblas_int         offsetCin,
                                         rocblas_int         ldc,
                                         rocblas_stride      stride_c,
                                         void*               d,
                                         rocblas_datatype    d_type,
                                         rocblas_int         offsetDin,
                                         rocblas_int         ldd,
                                         rocblas_stride      stride_d,
                                         rocblas_int         batch_count,
                                         rocblas_computetype compute_type,
                                         uint32_t            flags)
{
    if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
    {
        if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_f8, float>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_bf8, float>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_bf8, float>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_f8, float>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, float, rocblas_bf8, float>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f32_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, float, float>(EX_TEMPLATE_PARM);
    }
    else if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
    {
        if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_f8, rocblas_half>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_bf8, rocblas_half>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_bf8, rocblas_half>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_f8, rocblas_half>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_half, rocblas_half, rocblas_half>(EX_TEMPLATE_PARM);
    }
    else if(c_type == rocblas_datatype_f8_r && d_type == rocblas_datatype_f8_r)
    {
        if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_f8, rocblas_f8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f32_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, float, rocblas_f8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, float, rocblas_f8, rocblas_f8>(EX_TEMPLATE_PARM);
    }
    else if(c_type == rocblas_datatype_bf8_r && d_type == rocblas_datatype_bf8_r)
    {
        if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_bf8, rocblas_bf8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_f8, rocblas_bf8, rocblas_bf8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, rocblas_f8, rocblas_bf8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_bf8_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, float, rocblas_bf8, rocblas_bf8>(EX_TEMPLATE_PARM);
        else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f32_r)
            return rocblas_gemm_batched_ex3_typecasting<BATCHED, rocblas_bf8, float, rocblas_bf8>(EX_TEMPLATE_PARM);
    }

    return rocblas_status_not_implemented;
}

#undef EX_TEMPLATE_PARM

// Copy alpha and beta to host if on device
template <typename T>
rocblas_status rocblas_copy_alpha_beta_to_host_if_on_device(rocblas_handle      handle,
                                                            const T*&           alpha,
                                                            const T*&           beta,
                                                            rocblas_union_t&    alpha_h,
                                                            rocblas_union_t&    beta_h,
                                                            rocblas_int         k,
                                                            rocblas_computetype compute_type)
{
    switch(compute_type)
    {
    case rocblas_compute_type_f32:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_compute_type_f8_f8_f32:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_compute_type_f8_bf8_f32:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_compute_type_bf8_f8_f32:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_compute_type_bf8_bf8_f32:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    default:
        return rocblas_status_not_implemented;
    }
}

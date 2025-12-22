/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"
#include "rocblas_syrk_ex.hpp"

#include "../blas3/herk_syrk_device.hpp"
#include "../blas3/rocblas_syrk_herk.hpp"
#include "../blas3/rocblas_syrk_imp.hpp"

// templated alpha, beta, restricted n, k
template <typename API_INT,
          typename Tex,
          typename T,
          typename Tc,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          int  alpha,
          int  beta,
          bool HERM,
          char trans_A,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
rocblas_syrkx_herkx_ex_restricted_kernel(rocblas_int    N,
                                         API_INT        K,
                                         TConstPtr*     A,
                                         API_INT        lda,
                                         rocblas_stride stride_A,
                                         TConstPtr*     B,
                                         API_INT        ldb,
                                         rocblas_stride stride_B,
                                         TPtr*          C,
                                         API_INT        ldc,
                                         rocblas_stride stride_C,
                                         rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(A, blz, 0, stride_A);
    auto* dB = load_ptr_batch(B, blz, 0, stride_B);
    auto* dC = load_ptr_batch(C, blz, 0, stride_C);

    __shared__ Tex sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ Tex sB[BLK_N][BLK_K]; // shared memory for B
    Tex            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    size_t coord_A, coord_B;
    if(trans_A == 'N')
        coord_A = (blx * BLK_N + thxA) + thyA * size_t(lda);
    if(trans_A == 'T')
        coord_A = (blx * BLK_N + thxA) * size_t(lda) + thyA;
    if(trans_A == 'C')
        coord_A = (blx * BLK_N + thxA) * size_t(lda) + thyA;
    if(trans_A == 'C')
        coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;
    if(trans_A == 'T')
        coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;
    if(trans_A == 'N')
        coord_B = (bly * BLK_N + thyB) + thxB * size_t(ldb);

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = Tex(0.0);

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        sA[thyA][thxA] = conj_if_true < HERM && trans_A == 'C' > (Tex(dA[coord_A]));
        sB[thyB][thxB] = conj_if_true < HERM && trans_A == 'N' > (Tex(dB[coord_B]));

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        if(trans_A == 'N')
            coord_A += BLK_K * size_t(lda);
        if(trans_A == 'T')
            coord_A += BLK_K;
        if(trans_A == 'C')
            coord_A += BLK_K;

        if(trans_A == 'C')
            coord_B += BLK_K;
        if(trans_A == 'T')
            coord_B += BLK_K;
        if(trans_A == 'N')
            coord_B += BLK_K * size_t(ldb);
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;

            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(alpha == 1 && beta == 1)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = Tc(Tex(dC[coord_dCn * size_t(ldc) + coord_dCm]) + rC[n][m]);
                if(alpha == 1 && beta == -1)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = Tc(Tex(-dC[coord_dCn * size_t(ldc) + coord_dCm]) + rC[n][m]);
                if(alpha == -1 && beta == 0)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(-rC[n][m]);
                if(alpha == 1 && beta == 0)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(rC[n][m]);

                // Zero out imaginary part of diagonal if HERM
                if(HERM && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

// large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
// general alpha, beta, restricted n, k
template <typename API_INT,
          typename Tex,
          typename T,
          typename Tc,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          bool BETA_EQ_ZERO,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
rocblas_syrkx_herkx_ex_restricted_general_kernel(rocblas_int    N,
                                                 API_INT        K,
                                                 const Tex      alpha,
                                                 TConstPtr*     dA_array,
                                                 API_INT        lda,
                                                 rocblas_stride stride_a,
                                                 TConstPtr*     dB_array,
                                                 API_INT        ldb,
                                                 rocblas_stride stride_b,
                                                 const Tex      beta,
                                                 TPtr*          dC_array,
                                                 API_INT        ldc,
                                                 rocblas_stride stride_c,
                                                 rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ Tex sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ Tex sB[BLK_N][BLK_K]; // shared memory for B
    Tex            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = Tex(0.0);

    size_t coord_A, coord_B;
    if(TRANS == 'N')
        coord_A = (thxA + blx * BLK_N) + (thyA)*size_t(lda);
    if(TRANS == 'T')
        coord_A = (thxA + blx * BLK_N) * size_t(lda) + (thyA);
    if(TRANS == 'C')
        coord_A = (thxA + blx * BLK_N) * size_t(lda) + (thyA);

    if(TRANS == 'C')
        coord_B = thxB + (bly * BLK_N + thyB) * size_t(ldb);
    if(TRANS == 'T')
        coord_B = thxB + (bly * BLK_N + thyB) * size_t(ldb);
    if(TRANS == 'N')
        coord_B = thxB * size_t(ldb) + (bly * BLK_N + thyB);

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        sA[thyA][thxA] = conj_if_true < HERK && TRANS == 'C' > (Tex(dA[coord_A]));
        sB[thyB][thxB] = conj_if_true < HERK && TRANS == 'N' > (Tex(dB[coord_B]));

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        if(TRANS == 'N')
            coord_A += BLK_K * size_t(lda);
        if(TRANS == 'T')
            coord_A += BLK_K;
        if(TRANS == 'C')
            coord_A += BLK_K;

        if(TRANS == 'C')
            coord_B += BLK_K;
        if(TRANS == 'T')
            coord_B += BLK_K;
        if(TRANS == 'N')
            coord_B += BLK_K * size_t(ldb);
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;
            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(BETA_EQ_ZERO)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(alpha * rC[n][m]);
                else
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(
                        alpha * rC[n][m] + beta * Tex(dC[coord_dCn * size_t(ldc) + coord_dCm]));

                // Zero out imaginary part of diagonal if herk
                if(HERK && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

// syrkx/herkx name to show original source
// general alpha, beta, m, n, k
template <typename API_INT,
          typename Tex,
          typename T,
          typename Tc,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          bool BETA_EQ_ZERO,
          bool HERM,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
rocblas_syrkx_herkx_ex_general_kernel(rocblas_int    N,
                                      API_INT        K,
                                      const Tex      alpha,
                                      TConstPtr*     A,
                                      API_INT        lda,
                                      rocblas_stride stride_A,
                                      TConstPtr*     B,
                                      API_INT        ldb,
                                      rocblas_stride stride_B,
                                      const Tex      beta,
                                      TPtr*          C,
                                      API_INT        ldc,
                                      rocblas_stride stride_C,
                                      rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(A, blz, 0, stride_A);
    auto* dB = load_ptr_batch(B, blz, 0, stride_B);
    auto* dC = load_ptr_batch(C, blz, 0, stride_C);

    __shared__ Tex sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ Tex sB[BLK_N][BLK_K]; // shared memory for B
    Tex            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    int a_i_offset = thxA + BLK_N * blx;
    int a_j_offset = thyA;
    int b_i_offset = thxB;
    int b_j_offset = thyB + BLK_N * bly;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = Tex(0);

    API_INT kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        API_INT i = a_i_offset;
        API_INT j = kk + a_j_offset;
        if(i < N && j < K)
        {
            if(TRANS == 'N')
                sA[thyA][thxA] = Tex(dA[i + j * size_t(lda)]);
            if(TRANS == 'T')
                sA[thyA][thxA] = Tex(dA[i * size_t(lda) + j]);
            if(TRANS == 'C')
                sA[thyA][thxA] = conj_if_true<HERM>(Tex(dA[i * size_t(lda) + j]));
        }
        else
        {
            sA[thyA][thxA] = Tex(0);
        }
        i = kk + b_i_offset;
        j = b_j_offset;
        if(i < K && j < N)
        {
            if(TRANS == 'C')
                sB[thyB][thxB] = Tex(dB[i + j * size_t(ldb)]);
            if(TRANS == 'T')
                sB[thyB][thxB] = Tex(dB[i + j * size_t(ldb)]);
            if(TRANS == 'N')
                sB[thyB][thxB] = conj_if_true<HERM>(Tex(dB[i * size_t(ldb) + j]));
        }
        else
        {
            sB[thyB][thxB] = Tex(0);
        }

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        int coord_dCn = bly * BLK_N + n * DIM_N + thy;
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(BETA_EQ_ZERO)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(alpha * rC[n][m]);
                else
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = Tc(
                        alpha * rC[n][m] + beta * Tex(dC[coord_dCn * size_t(ldc) + coord_dCm]));

                // Zero out imaginary part of diagonal if HERM
                if(HERM && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

template <typename Tex, typename Ta, typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_ex_template(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_A,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const Tex*        alpha_in,
                                      const Ta*         A,
                                      rocblas_datatype  A_type,
                                      rocblas_stride    offset_A,
                                      rocblas_int       lda,
                                      rocblas_stride    stride_A,
                                      const Tex*        beta_in,
                                      T*                C,
                                      rocblas_datatype  C_type,
                                      rocblas_stride    offset_C,
                                      rocblas_int       ldc,
                                      rocblas_stride    stride_C,
                                      rocblas_datatype  compute_type,
                                      rocblas_int       batch_count)
{

    constexpr bool BATCHED = false;
    constexpr bool HERM    = false;

    if constexpr(sizeof(Tex) < 8)
    {
        constexpr bool FORCEDGEMM = true;

        size_t size
            = rocblas_internal_syrk_herk_workspace<T, FORCEDGEMM>(handle, n, k, batch_count);

        //Allocate Workspace memory
        auto w_mem = handle->device_malloc(size);
        if(!w_mem)
            return rocblas_status_memory_error;

        hipStream_t rocblas_stream = handle->get_stream();

        // Note: alpha and beta always copied over to host by now
        if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
            return rocblas_status_success;

        bool a_calc_invalid = !alpha_in || (*alpha_in != 0 && (!A));
        if(!C || (k && a_calc_invalid))
            return rocblas_status_invalid_pointer;

        // upgrade to complex if needed
        // TODO: Graph safety?
        const Tex alpha_val = (Tex)(*alpha_in);
        const Tex beta_val  = (Tex)(*beta_in);

        const Tex* alpha = &alpha_val;
        const Tex* beta  = &beta_val;

        rocblas_operation trans_orig
            = rocblas_operation_none == trans_A
                  ? rocblas_operation_none
                  : (HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
        rocblas_operation trans_opp
            = rocblas_operation_none == trans_A
                  ? (HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose)
                  : rocblas_operation_none;

        // Launch kernel to copy the data from triangular matrix to the workspace memory
        if(rocblas_fill_upper == uplo)
            RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<true, true, HERM>(
                handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));
        else
            RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<true, false, HERM>(
                handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_gemm_ex_template<BATCHED>(handle,
                                                                   trans_orig,
                                                                   trans_opp,
                                                                   n,
                                                                   n,
                                                                   k,
                                                                   alpha,
                                                                   A,
                                                                   A_type,
                                                                   offset_A,
                                                                   lda,
                                                                   stride_A,
                                                                   A,
                                                                   A_type,
                                                                   offset_A,
                                                                   lda,
                                                                   stride_A,
                                                                   beta,
                                                                   C,
                                                                   C_type,
                                                                   offset_C,
                                                                   ldc,
                                                                   stride_C,
                                                                   C,
                                                                   C_type,
                                                                   offset_C,
                                                                   ldc,
                                                                   stride_C,
                                                                   batch_count,
                                                                   compute_type,
                                                                   rocblas_gemm_algo_standard,
                                                                   0,
                                                                   0)));

        // Launch kernel to copy the data from workspace memory back to triangular matrix
        if(rocblas_fill_upper == uplo)
            RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<false, true, HERM>(
                handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));
        else
            RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<false, false, HERM>(
                handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));

        return rocblas_status_success;
    }
    else
    {
        using API_INT = rocblas_int;

        hipStream_t stream = handle->get_stream();

        // Note: alpha and beta always copied over to host by now
        if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
            return rocblas_status_success;

        bool a_calc_invalid = !alpha_in || (*alpha_in != 0 && (!A));
        if(!C || (k && a_calc_invalid))
            return rocblas_status_invalid_pointer;

        int batches = handle->getBatchGridDim((int)batch_count);

        // syrkx has same behavior for alpha == 0 and k == 0. Special code is needed
        // for alpha == 0, no special code is needed for k == 0. It is more efficient
        // setting k = 0 than adding extra code to a kernel to handle alpha == 0
        if(*alpha_in == 0)
            k = 0;

        const Tex alpha = (Tex)(*alpha_in);
        const Tex beta  = (Tex)(*beta_in);

        if((n % 32 == 0) && (k % 8 == 0)) // restricted kernels
        {
            // Can also be used with:
            // n is mult of 64, k is mult of 4
            // const int dim_n = 16;
            // const int blk_n = 64;
            // const int blk_k = 4;

            // n is mult of 32, k is mult of 8
            const int dim_n = 16;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_n, dim_n, 1);
            dim3      dimGrid(n / blk_n, n / blk_n, batches);

            // clang-format off
            if(alpha == Tex(1.0) && beta == Tex(1.0))
            {
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tex(1.0) && beta == Tex(-1.0))
            {
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tex(1.0) && beta == Tex(0.0))
            {
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tex(-1.0) && beta == Tex(0.0))
            {
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                        <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'N', 'U'>),
                        dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(beta == Tex(0))
            {
                // general alpha; beta == 0
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            else
            {
                // general alpha, beta
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            // clang-format on
        }
        else
        {
            const int dim_n = 16;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_n, dim_n, 1);
            dim3      dimGrid(((n - 1) / blk_n) + 1, ((n - 1) / blk_n) + 1, batches);

            // clang-format off
            if(beta == Tex(0))
            {
                // general n, k, alpha; beta == 0
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            else
            {
                // general n, k, alpha, beta
                if((rocblas_operation_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'T', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'T', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            // clang-format on
        }

        return rocblas_status_success;
    }
}

template <bool BATCHED, typename T, typename U, typename Tex>
rocblas_status rocblas_syrk_ex_typecasting(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_A,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const void*       alpha,
                                           const void*       A,
                                           rocblas_datatype  A_type,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           const void*       beta,
                                           void*             C,
                                           rocblas_datatype  C_type,
                                           rocblas_stride    offset_C,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_C,
                                           rocblas_datatype  compute_type,
                                           rocblas_int       batch_count)
{
    auto check_numerics = handle->check_numerics;

    rocblas_status status = rocblas_syrk_arg_check<rocblas_int>(handle,
                                                                uplo,
                                                                trans_A,
                                                                n,
                                                                k,
                                                                (Tex*)alpha,
                                                                (const T*)A,
                                                                offset_A,
                                                                lda,
                                                                stride_A,
                                                                (Tex*)beta,
                                                                (U*)C,
                                                                offset_C,
                                                                ldc,
                                                                stride_C,
                                                                batch_count);
    if(status != rocblas_status_continue)
        return status;

    if(!BATCHED)
    {
        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_syrk_strided_batched_ex" : "rocblas_syrk_ex";
            bool is_input = true;
            status        = rocblas_herk_syrk_check_numerics<false>(check_numerics_string,
                                                             handle,
                                                             uplo,
                                                             trans_A,
                                                             n,
                                                             k,
                                                             (const T*)A,
                                                             lda,
                                                             stride_A,
                                                             (U*)C,
                                                             ldc,
                                                             stride_C,
                                                             batch_count,
                                                             check_numerics,
                                                             is_input);
            if(status != rocblas_status_success)
                return status;
        }

        status = rocblas_internal_syrk_ex_template<Tex>(handle,
                                                        uplo,
                                                        trans_A,
                                                        n,
                                                        k,
                                                        (const Tex*)alpha,
                                                        (const T*)A,
                                                        A_type,
                                                        offset_A,
                                                        lda,
                                                        stride_A,
                                                        (const Tex*)beta,
                                                        (U*)C,
                                                        C_type,
                                                        offset_C,
                                                        ldc,
                                                        stride_C,
                                                        compute_type,
                                                        1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_syrk_strided_batched_ex" : "rocblas_syrk_ex";
            bool is_input = false;
            status        = rocblas_herk_syrk_check_numerics<false>(check_numerics_string,
                                                             handle,
                                                             uplo,
                                                             trans_A,
                                                             n,
                                                             k,
                                                             (const T*)A,
                                                             lda,
                                                             stride_A,
                                                             (U*)C,
                                                             ldc,
                                                             stride_C,
                                                             batch_count,
                                                             check_numerics,
                                                             is_input);
            if(status != rocblas_status_success)
                return status;
        }
        return rocblas_status_success;
    }
    else
        return rocblas_status_not_implemented;
}

template <bool BATCHED>
rocblas_status rocblas_syrk_ex_template(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation trans_A,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const void*       alpha,
                                        const void*       A,
                                        rocblas_datatype  A_type,
                                        rocblas_stride    offset_A,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_A,
                                        const void*       beta,
                                        void*             C,
                                        rocblas_datatype  C_type,
                                        rocblas_stride    offset_C,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_C,
                                        rocblas_datatype  compute_type,
                                        rocblas_int       batch_count)
{
    if(!n || !batch_count)
        return rocblas_status_success;

    rocblas_status status = rocblas_status_not_implemented;

#define SYRK_EX_TYPECASTING_PARAM                                                            \
    handle, uplo, trans_A, n, k, alpha, A, A_type, offset_A, lda, stride_A, beta, C, C_type, \
        offset_C, ldc, stride_C, compute_type, batch_count

    if(A_type == rocblas_datatype_bf16_r && C_type == rocblas_datatype_bf16_r
       && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_bf16_r && C_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_bfloat16, float, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f16_r && C_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f16_r && C_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_half, float, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f32_r && C_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f64_r)
        status
            = rocblas_syrk_ex_typecasting<BATCHED, float, float, double>(SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f32_r && C_type == rocblas_datatype_f64_r
            && compute_type == rocblas_datatype_f64_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, float, double, double>(
            SYRK_EX_TYPECASTING_PARAM);
    else
        status = rocblas_status_not_implemented;

#undef SYRK_EX_TYPECASTING_PARAM

    return status;
}

#ifdef INSTANTIATE_SYRK_EX_TEMPLATE
#error INSTANTIATE_SYRK_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_SYRK_EX_TEMPLATE(BATCHED)                                                 \
    template rocblas_status rocblas_syrk_ex_template<BATCHED>(rocblas_handle    handle,       \
                                                              rocblas_fill      uplo,         \
                                                              rocblas_operation trans_A,      \
                                                              rocblas_int       n,            \
                                                              rocblas_int       k,            \
                                                              const void*       alpha,        \
                                                              const void*       A,            \
                                                              rocblas_datatype  A_type,       \
                                                              rocblas_stride    offset_A,     \
                                                              rocblas_int       lda,          \
                                                              rocblas_stride    stride_A,     \
                                                              const void*       beta,         \
                                                              void*             C,            \
                                                              rocblas_datatype  C_type,       \
                                                              rocblas_stride    offset_C,     \
                                                              rocblas_int       ldc,          \
                                                              rocblas_stride    stride_C,     \
                                                              rocblas_datatype  compute_type, \
                                                              rocblas_int       batch_count);

INSTANTIATE_SYRK_EX_TEMPLATE(false)
// INSTANTIATE_SYRK_EX_TEMPLATE(true)

#undef INSTANTIATE_SYRK_EX_TEMPLATE

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
#pragma once

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
    int      thx   = threadIdx.x; // thread's m position in C
    int      thy   = threadIdx.y; // thread's n position in C
    int      idt   = DIM_N * thy + thx; // thread's number
    int      blx   = blockIdx.x; // block's m position
    int      bly   = blockIdx.y; // block's n position
    uint32_t batch = blockIdx.z; // block's matrix in the batch

    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(A, batch, 0, stride_A);
    auto* dB = load_ptr_batch(B, batch, 0, stride_B);
    auto* dC = load_ptr_batch(C, batch, 0, stride_C);

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

// general alpha, beta, restricted n, k
template <typename API_INT,
          typename Tex,
          typename Tab_ex,
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
                                                 const Tab_ex   alpha,
                                                 TConstPtr*     dA_array,
                                                 API_INT        lda,
                                                 rocblas_stride stride_a,
                                                 TConstPtr*     dB_array,
                                                 API_INT        ldb,
                                                 rocblas_stride stride_b,
                                                 const Tab_ex   beta,
                                                 TPtr*          dC_array,
                                                 API_INT        ldc,
                                                 rocblas_stride stride_c,
                                                 rocblas_int    batch_count)
{
    int      thx   = threadIdx.x; // thread's m position in C
    int      thy   = threadIdx.y; // thread's n position in C
    int      idt   = DIM_N * thy + thx; // thread's number
    int      blx   = blockIdx.x; // block's m position
    int      bly   = blockIdx.y; // block's n position
    uint32_t batch = blockIdx.z; // block's matrix in the batch

    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, batch, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, batch, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, batch, 0, stride_c);

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
          typename Tab_ex,
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
                                      const Tab_ex   alpha,
                                      TConstPtr*     A,
                                      API_INT        lda,
                                      rocblas_stride stride_A,
                                      TConstPtr*     B,
                                      API_INT        ldb,
                                      rocblas_stride stride_B,
                                      const Tab_ex   beta,
                                      TPtr*          C,
                                      API_INT        ldc,
                                      rocblas_stride stride_C,
                                      rocblas_int    batch_count)
{
    int      thx   = threadIdx.x; // thread's m position in C
    int      thy   = threadIdx.y; // thread's n position in C
    int      idt   = DIM_N * thy + thx; // thread's number
    int      blx   = blockIdx.x; // block's m position
    int      bly   = blockIdx.y; // block's n position
    uint32_t batch = blockIdx.z; // block's matrix in the batch

    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(A, batch, 0, stride_A);
    auto* dB = load_ptr_batch(B, batch, 0, stride_B);
    auto* dC = load_ptr_batch(C, batch, 0, stride_C);

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

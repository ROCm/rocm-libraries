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

#include "handle.hpp"

namespace
{
    /*
     * Copies data from d_a into a, and d_b into b. Intended to copy
     * global memory into local memory for the kernel.
     */
    template <bool        TRANSA,
              bool        TRANSB,
              rocblas_int BUFA_N,
              rocblas_int BUFA_M,
              rocblas_int BUFB_N,
              rocblas_int BUFB_M,
              rocblas_int DIM_N_A,
              rocblas_int DIM_M_A,
              rocblas_int DIM_N_B,
              rocblas_int DIM_M_B,
              bool        ALPHA_ONE,
              bool        MINPLUS,
              bool        BOUNDS,
              typename T,
              typename U>
    __device__ void global_to_local(T        a[BUFA_M][BUFA_N],
                                    T        b[BUFB_M][BUFB_N],
                                    T        alpha,
                                    U const* d_a,
                                    size_t   lda,
                                    int      a_i,
                                    int      a_j,
                                    int      idx_a,
                                    int      idy_a,
                                    U const* d_b,
                                    size_t   ldb,
                                    int      b_i,
                                    int      b_j,
                                    int      idx_b,
                                    int      idy_b,
                                    int      m,
                                    int      n,
                                    int      k)
    {
        bool out_of_bounds = false;
        for(int j = 0; j < BUFA_N; ++j)
        {
            for(int i = 0; i < BUFA_M; ++i)
            {

                int ai = a_i + i * DIM_M_A + idx_a;
                int aj = a_j + j * DIM_N_A + idy_a;

                if(BOUNDS)
                {
                    if(!TRANSA)
                    {
                        out_of_bounds = ai >= m || aj >= k;
                        ai            = min(ai, m - 1);
                        aj            = min(aj, k - 1);
                    }
                    else
                    {
                        out_of_bounds = ai >= k || aj >= m;
                        ai            = min(ai, k - 1);
                        aj            = min(aj, m - 1);
                    }
                }

                if(BOUNDS && out_of_bounds)
                {
                    if constexpr(MINPLUS)
                        rocblas_set_max_value(a[i][j]);
                    else
                        a[i][j] = 0;
                }
                else if(ALPHA_ONE)
                    a[i][j] = d_a[aj * lda + ai];
                else
                    a[i][j] = alpha ? alpha * d_a[aj * lda + ai] : 0;
            }
        }

        for(int j = 0; j < BUFB_N; ++j)
        {
            for(int i = 0; i < BUFB_M; ++i)
            {

                int bi = b_i + i * DIM_M_B + idx_b;
                int bj = b_j + j * DIM_N_B + idy_b;

                if(BOUNDS)
                {
                    if(!TRANSB)
                    {
                        out_of_bounds = bi >= k || bj >= n;
                        bi            = min(bi, k - 1);
                        bj            = min(bj, n - 1);
                    }
                    else
                    {
                        out_of_bounds = bi >= n || bj >= k;
                        bi            = min(bi, n - 1);
                        bj            = min(bj, k - 1);
                    }
                }

                if(BOUNDS && out_of_bounds)
                {
                    if constexpr(MINPLUS)
                        rocblas_set_max_value(b[i][j]);
                    else
                        b[i][j] = 0;
                }
                else if(ALPHA_ONE)
                    b[i][j] = d_b[bj * ldb + bi];
                else
                    b[i][j] = alpha ? alpha * d_b[bj * ldb + bi] : 0;
            }
        }
    }

    template <typename T, typename U, std::enable_if_t<!rocblas_is_array2<T>, int> = 0>
    __device__ void vector2_min(const T& c_in, U& c_out)
    {
        c_out = c_in;
    }

    template <typename T, typename U, std::enable_if_t<rocblas_is_array2<T>, int> = 0>
    __device__ void vector2_min(const T& c_in, U& c_out)
    {
        c_out = fminf(c_in.x, c_in.y);
    }

    template <typename T, typename U, std::enable_if_t<!rocblas_is_array2<T>, int> = 0>
    __device__ void vector2_add(const T& c_in, U& c_out)
    {
        c_out = c_in;
    }

    template <typename T, typename U, std::enable_if_t<rocblas_is_array2<T>, int> = 0>
    __device__ void vector2_add(const T& c_in, U& c_out)
    {
        c_out = c_in.x + c_in.y;
    }

    /*
     * Copies data from c into d_c, intended to copy local memory back to global memory.
     */
    template <rocblas_int THR_N,
              rocblas_int THR_M,
              rocblas_int DIM_N,
              rocblas_int DIM_M,
              bool        MINPLUS,
              bool        BOUNDS,
              typename T,
              typename U>
    __device__ void local_to_global(const T* d_c,
                                    T*       d_d,
                                    size_t   ldc,
                                    size_t   ldd,
                                    T        beta,
                                    int      c_i,
                                    int      c_j,
                                    int      idx,
                                    int      idy,
                                    U        c[THR_N][THR_M],
                                    int      m,
                                    int      n)
    {
        for(int j = 0; j < THR_N; ++j)
        {
            for(int i = 0; i < THR_M; ++i)
            {
                int ci = c_i + i * DIM_M + idx;
                int cj = c_j + j * DIM_N + idy;

                T c_min;
                if constexpr(MINPLUS)
                    vector2_min(c[j][i], c_min);
                else
                    vector2_add(c[j][i], c_min);

                if(BOUNDS)
                {
                    if(ci < m && cj < n)
                    {
                        auto dc_scaled = beta ? beta * d_c[cj * ldc + ci] : 0;
                        if constexpr(MINPLUS)
                            d_d[cj * ldd + ci] = fminf(dc_scaled, c_min);
                        else
                            d_d[cj * ldd + ci] = dc_scaled + c_min;
                    }
                }
                else
                {
                    auto dc_scaled = beta ? beta * d_c[cj * ldc + ci] : 0;
                    if constexpr(MINPLUS)
                        d_d[cj * ldd + ci] = fminf(dc_scaled, c_min);
                    else
                        d_d[cj * ldd + ci] = dc_scaled + c_min;
                }
            }
        }
    }

    /*
     * Copies data from a to s_a, and b to s_b.
     */
    template <bool        TRANSA,
              bool        TRANSB,
              rocblas_int BLK_N,
              rocblas_int BLK_M,
              rocblas_int BLK_K,
              rocblas_int BUFA_N,
              rocblas_int BUFA_M,
              rocblas_int BUFB_N,
              rocblas_int BUFB_M,
              rocblas_int DIM_N_A,
              rocblas_int DIM_M_A,
              rocblas_int DIM_N_B,
              rocblas_int DIM_M_B,
              typename T,
              typename U>
    __device__ void local_to_shared(__shared__ T s_a[BLK_M][BLK_K],
                                    __shared__ T s_b[BLK_N][BLK_K],
                                    U            a[BUFA_M][BUFA_N],
                                    U            b[BUFB_M][BUFB_N],
                                    int          idx_a,
                                    int          idy_a,
                                    int          idx_b,
                                    int          idy_b)
    {
        for(int j = 0; j < BUFA_N; ++j)
            for(int i = 0; i < BUFA_M; ++i)
            {
                if(!TRANSA)
                    s_a[i * DIM_M_A + idx_a][j * DIM_N_A + idy_a] = a[i][j];
                else
                    s_a[j * DIM_N_A + idy_a][i * DIM_M_A + idx_a] = a[i][j];
            }

        for(int j = 0; j < BUFB_N; ++j)
            for(int i = 0; i < BUFB_M; ++i)
            {
                if(!TRANSB)
                    s_b[j * DIM_N_B + idy_b][i * DIM_M_B + idx_b] = b[i][j];
                else
                    s_b[i * DIM_M_B + idx_b][j * DIM_N_B + idy_b] = b[i][j];
            }
    }

    /*
     * Copies data from s_a to a, and s_b to b.
     */
    template <rocblas_int THR_N,
              rocblas_int THR_M,
              rocblas_int BLK_N,
              rocblas_int BLK_M,
              rocblas_int BLK_K,
              rocblas_int DIM_N,
              rocblas_int DIM_M,
              typename T,
              typename T2,
              std::enable_if_t<!rocblas_is_array2<T2>, int> = 0>
    __device__ void shared_to_local(T2           a[THR_M],
                                    T2           b[THR_N],
                                    __shared__ T s_a[BLK_M][BLK_K],
                                    __shared__ T s_b[BLK_N][BLK_K],
                                    int          k,
                                    int          idx,
                                    int          idy)
    {
        for(int i = 0; i < THR_M; i++)
            a[i] = s_a[i * DIM_M + idx][k];

        for(int j = 0; j < THR_N; j++)
            b[j] = s_b[j * DIM_N + idy][k];
    }

    template <rocblas_int THR_N,
              rocblas_int THR_M,
              rocblas_int BLK_N,
              rocblas_int BLK_M,
              rocblas_int BLK_K,
              rocblas_int DIM_N,
              rocblas_int DIM_M,
              typename T,
              typename T2,
              std::enable_if_t<rocblas_is_array2<T2>, int> = 0>
    __device__ void shared_to_local(T2           a[THR_M],
                                    T2           b[THR_N],
                                    __shared__ T s_a[BLK_M][BLK_K],
                                    __shared__ T s_b[BLK_N][BLK_K],
                                    int          k,
                                    int          idx,
                                    int          idy)
    {
        for(int i = 0; i < THR_M; i++)
        {
            a[i].x = s_a[i * DIM_M + idx][k];
            a[i].y = s_a[i * DIM_M + idx][k + 1];
        }

        for(int j = 0; j < THR_N; j++)
        {
            b[j].x = s_b[j * DIM_N + idy][k];
            b[j].y = s_b[j * DIM_N + idy][k + 1];
        }
    }

    template <rocblas_int THR_N,
              rocblas_int THR_M,
              bool        MINPLUS,
              typename T,
              std::enable_if_t<!rocblas_is_array2<T>, int> = 0>
    __device__ void initialize_local_output(T c[THR_N][THR_M])
    {
        for(int j = 0; j < THR_N; j++)
            for(int i = 0; i < THR_M; i++)
            {
                if constexpr(MINPLUS)
                    rocblas_set_max_value(c[j][i]);
                else
                    c[j][i] = 0;
            }
    }

    template <rocblas_int THR_N,
              rocblas_int THR_M,
              bool        MINPLUS,
              typename T,
              std::enable_if_t<rocblas_is_array2<T>, int> = 0>
    __device__ void initialize_local_output(T c[THR_N][THR_M])
    {
        for(int j = 0; j < THR_N; j++)
            for(int i = 0; i < THR_M; i++)
            {
                if constexpr(MINPLUS)
                {
                    auto tmp = c[j][i].x;
                    rocblas_set_max_value(tmp);
                    c[j][i].x = tmp;
                    c[j][i].y = tmp;
                }
                else
                {
                    c[j][i].x = 0;
                    c[j][i].y = 0;
                }
            }
    }

    template <typename T, typename T2>
    __device__ void rocblas_geam_ex_min3(const T2& a, T& c)
    {
        c = fminf(fminf(a.x, a.y), c);
    }

    template <>
    __device__ void rocblas_geam_ex_min3(const rocblas_half2& a, rocblas_half2& c)
    {
        // asm volatile("v_pk_min_f16 %0, %1, %2;" : "=v"(c) : "v"(c), "v"(a));
        c.x = fminf(c.x, a.x);
        c.y = fminf(c.y, a.y);
    }

    template <>
    __device__ void rocblas_geam_ex_min3(const double& a, double& c)
    {
        // asm volatile("v_min_f64 %0, %1, %2;" : "=v"(c) : "v"(c), "v"(a));
        c = fmin(a, c);
    }

    template <>
    __device__ void rocblas_geam_ex_min3(const float2& a, float& c)
    {
        c = fminf(fminf(a.x, a.y), c);
    }

    template <typename T, typename T2>
    __device__ void rocblas_geam_ex_minadd3(const T2& a, const T2& b, T& c)
    {
        T2 c2;
        c2.x = fminf(a.x, b.x);
        c2.y = fminf(a.y, b.y);
        c += c2;
    }

    template <>
    __device__ void rocblas_geam_ex_minadd3(const double& a, const double& b, double& c)
    {
        // asm volatile("v_min_f64 %0, %1, %2;" : "=v"(tmp) : "v"(b), "v"(a));
        c += fmin(a, b);
    }

    /*
     * Computes the "minplus" operation Cij = min(Aik + Bkj, Cij)
     */
    template <rocblas_int THR_N, rocblas_int THR_M, typename T, typename T2>
    __device__ void compute_minplus(T2 a[THR_M], T2 b[THR_N], T c[THR_N][THR_M])
    {
        for(int j = 0; j < THR_N; j++)
            for(int i = 0; i < THR_M; i++)
                rocblas_geam_ex_min3(a[i] + b[j], c[j][i]);
    }

    /*
     * Computes the "plusmin" operation Cij = min(Aik, Bkj) + Cij
     */
    template <rocblas_int THR_N, rocblas_int THR_M, typename T, typename T2>
    __device__ void compute_plusmin(T2 a[THR_M], T2 b[THR_N], T c[THR_N][THR_M])
    {
        for(int j = 0; j < THR_N; j++)
            for(int i = 0; i < THR_M; i++)
                rocblas_geam_ex_minadd3(a[i], b[j], c[j][i]);
    }

    template <typename T,
              typename Tab,
              typename Tc,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              char TRANSA_C,
              char TRANSB_C,
              bool ALPHA_ONE,
              bool BOUNDS,
              bool MINPLUS,
              typename TScal,
              typename TConstPtr,
              typename TPtr>
    ROCBLAS_KERNEL(DIM_M* DIM_N)
    geam_min_plus_kernel(rocblas_int               M,
                         rocblas_int               N,
                         rocblas_int               K,
                         TScal                     alpha_in,
                         TConstPtr*                dA_input,
                         rocblas_int               lda,
                         rocblas_stride            a_st_or_of,
                         TConstPtr*                dB_input,
                         rocblas_int               ldb,
                         rocblas_stride            b_st_or_of,
                         TScal                     beta_in,
                         TConstPtr*                dC_input,
                         rocblas_int               ldc,
                         rocblas_stride            c_st_or_of,
                         TPtr*                     dD_input,
                         rocblas_int               ldd,
                         rocblas_stride            d_st_or_of,
                         rocblas_int               batch_count,
                         rocblas_geam_ex_operation geam_ex_op)
    {
        int   blz   = blockIdx.z; // block's matrix in the batch
        auto  alpha = load_scalar(alpha_in, blockIdx.z, 1);
        auto  beta  = load_scalar(beta_in, blockIdx.z, 1);
        auto* d_a   = alpha ? load_ptr_batch(dA_input, blz, a_st_or_of) : nullptr;
        auto* d_b   = alpha ? load_ptr_batch(dB_input, blz, b_st_or_of) : nullptr;
        auto* d_c   = beta ? load_ptr_batch(dC_input, blz, c_st_or_of) : nullptr;
        auto* d_d   = load_ptr_batch(dD_input, blz, d_st_or_of);

        constexpr bool TRANSA = TRANSA_C != 'N';
        constexpr bool TRANSB = TRANSB_C != 'N';

        constexpr int THR_M = BLK_M / DIM_M;
        constexpr int THR_N = BLK_N / DIM_N;

        constexpr int BUFA_M = TRANSA ? BLK_K / DIM_M_A : BLK_M / DIM_M_A;
        constexpr int BUFA_N = TRANSA ? BLK_M / DIM_N_A : BLK_K / DIM_N_A;
        constexpr int BUFB_M = TRANSB ? BLK_N / DIM_M_B : BLK_K / DIM_M_B;
        constexpr int BUFB_N = TRANSB ? BLK_K / DIM_N_B : BLK_N / DIM_N_B;

        int num_blocksx = (M - 1) / BLK_M + 1;
        int blx         = blockIdx.x % num_blocksx; // block's m position in C/D
        int bly         = blockIdx.x / num_blocksx; // block's n position in C/D

        int idx   = threadIdx.x; // thread's m position in C/D
        int idy   = threadIdx.y; // thread's n position in C/D
        int idt   = DIM_M * idy + idx; // thread's global number
        int idx_a = idt % DIM_M_A; // thread's m position for loading A
        int idy_a = idt / DIM_M_A; // thread's n position for loading A
        int idx_b = idt % DIM_M_B; // thread's m position for loading B
        int idy_b = idt / DIM_M_B; // thread's n position for loading B

        T a0[BUFA_M][BUFA_N]; // local array for loading A
        T a1[BUFA_M][BUFA_N]; // local array for loading A
        T b0[BUFB_M][BUFB_N]; // local array for loading B
        T b1[BUFB_M][BUFB_N]; // local array for loading B

        __shared__ T s_a0[BLK_M][BLK_K]; // shared memory for A
        __shared__ T s_a1[BLK_M][BLK_K]; // shared memory for A
        __shared__ T s_b0[BLK_N][BLK_K]; // shared memory for B
        __shared__ T s_b1[BLK_N][BLK_K]; // shared memory for B

        Tab a[THR_M]; // input local array A
        Tab b[THR_N]; // input local array B
        Tc  c[THR_N][THR_M]; // output local array C
        int ai = 0;
        int aj = 0;
        int bi = 0;
        int bj = 0;

        if(TRANSA)
            aj = blx * BLK_M;
        else
            ai = blx * BLK_M;

        if(TRANSB)
            bi = bly * BLK_N;
        else
            bj = bly * BLK_N;

        int ci = blx * BLK_M;
        int cj = bly * BLK_N;

        constexpr int k_add = rocblas_is_array2<Tab> ? 2 : 1;

        initialize_local_output<THR_N, THR_M, MINPLUS>(c);

        global_to_local<TRANSA,
                        TRANSB,
                        BUFA_N,
                        BUFA_M,
                        BUFB_N,
                        BUFB_M,
                        DIM_N_A,
                        DIM_M_A,
                        DIM_N_B,
                        DIM_M_B,
                        ALPHA_ONE,
                        MINPLUS,
                        BOUNDS>(
            a0, b0, alpha, d_a, lda, ai, aj, idx_a, idy_a, d_b, ldb, bi, bj, idx_b, idy_b, M, N, K);

        if(TRANSA)
            ai += BLK_K;
        else
            aj += BLK_K;
        if(TRANSB)
            bj += BLK_K;
        else
            bi += BLK_K;

        global_to_local<TRANSA,
                        TRANSB,
                        BUFA_N,
                        BUFA_M,
                        BUFB_N,
                        BUFB_M,
                        DIM_N_A,
                        DIM_M_A,
                        DIM_N_B,
                        DIM_M_B,
                        ALPHA_ONE,
                        MINPLUS,
                        BOUNDS>(
            a1, b1, alpha, d_a, lda, ai, aj, idx_a, idy_a, d_b, ldb, bi, bj, idx_b, idy_b, M, N, K);

        if(TRANSA)
            ai += BLK_K;
        else
            aj += BLK_K;
        if(TRANSB)
            bj += BLK_K;
        else
            bi += BLK_K;

        local_to_shared<TRANSA,
                        TRANSB,
                        BLK_N,
                        BLK_M,
                        BLK_K,
                        BUFA_N,
                        BUFA_M,
                        BUFB_N,
                        BUFB_M,
                        DIM_N_A,
                        DIM_M_A,
                        DIM_N_B,
                        DIM_M_B>(s_a0, s_b0, a0, b0, idx_a, idy_a, idx_b, idy_b);
        __syncthreads();

        for(int k1 = 0; k1 < BLK_K; k1 += k_add)
        {
            shared_to_local<THR_N, THR_M, BLK_N, BLK_M, BLK_K, DIM_N, DIM_M>(
                a, b, s_a0, s_b0, k1, idx, idy);
            if constexpr(MINPLUS)
                compute_minplus<THR_N, THR_M>(a, b, c);
            else
                compute_plusmin<THR_N, THR_M>(a, b, c);
        }

        local_to_shared<TRANSA,
                        TRANSB,
                        BLK_N,
                        BLK_M,
                        BLK_K,
                        BUFA_N,
                        BUFA_M,
                        BUFB_N,
                        BUFB_M,
                        DIM_N_A,
                        DIM_M_A,
                        DIM_N_B,
                        DIM_M_B>(s_a1, s_b1, a1, b1, idx_a, idy_a, idx_b, idy_b);
        __syncthreads();

        for(int l = 0; l < K - BLK_K - BLK_K; l += BLK_K + BLK_K)
        {
            global_to_local<TRANSA,
                            TRANSB,
                            BUFA_N,
                            BUFA_M,
                            BUFB_N,
                            BUFB_M,
                            DIM_N_A,
                            DIM_M_A,
                            DIM_N_B,
                            DIM_M_B,
                            ALPHA_ONE,
                            MINPLUS,
                            BOUNDS>(a0,
                                    b0,
                                    alpha,
                                    d_a,
                                    lda,
                                    ai,
                                    aj,
                                    idx_a,
                                    idy_a,
                                    d_b,
                                    ldb,
                                    bi,
                                    bj,
                                    idx_b,
                                    idy_b,
                                    M,
                                    N,
                                    K);

            if(TRANSA)
                ai += BLK_K;
            else
                aj += BLK_K;
            if(TRANSB)
                bj += BLK_K;
            else
                bi += BLK_K;

            for(int k1 = 0; k1 < BLK_K; k1 += k_add)
            {
                shared_to_local<THR_N, THR_M, BLK_N, BLK_M, BLK_K, DIM_N, DIM_M>(
                    a, b, s_a1, s_b1, k1, idx, idy);
                if constexpr(MINPLUS)
                    compute_minplus<THR_N, THR_M>(a, b, c);
                else
                    compute_plusmin<THR_N, THR_M>(a, b, c);
            }

            local_to_shared<TRANSA,
                            TRANSB,
                            BLK_N,
                            BLK_M,
                            BLK_K,
                            BUFA_N,
                            BUFA_M,
                            BUFB_N,
                            BUFB_M,
                            DIM_N_A,
                            DIM_M_A,
                            DIM_N_B,
                            DIM_M_B>(s_a0, s_b0, a0, b0, idx_a, idy_a, idx_b, idy_b);
            __syncthreads();

            global_to_local<TRANSA,
                            TRANSB,
                            BUFA_N,
                            BUFA_M,
                            BUFB_N,
                            BUFB_M,
                            DIM_N_A,
                            DIM_M_A,
                            DIM_N_B,
                            DIM_M_B,
                            ALPHA_ONE,
                            MINPLUS,
                            BOUNDS>(a1,
                                    b1,
                                    alpha,
                                    d_a,
                                    lda,
                                    ai,
                                    aj,
                                    idx_a,
                                    idy_a,
                                    d_b,
                                    ldb,
                                    bi,
                                    bj,
                                    idx_b,
                                    idy_b,
                                    M,
                                    N,
                                    K);
            if(TRANSA)
                ai += BLK_K;
            else
                aj += BLK_K;
            if(TRANSB)
                bj += BLK_K;
            else
                bi += BLK_K;

            for(int k1 = 0; k1 < BLK_K; k1 += k_add)
            {
                shared_to_local<THR_N, THR_M, BLK_N, BLK_M, BLK_K, DIM_N, DIM_M>(
                    a, b, s_a0, s_b0, k1, idx, idy);
                if constexpr(MINPLUS)
                    compute_minplus<THR_N, THR_M>(a, b, c);
                else
                    compute_plusmin<THR_N, THR_M>(a, b, c);
            }

            local_to_shared<TRANSA,
                            TRANSB,
                            BLK_N,
                            BLK_M,
                            BLK_K,
                            BUFA_N,
                            BUFA_M,
                            BUFB_N,
                            BUFB_M,
                            DIM_N_A,
                            DIM_M_A,
                            DIM_N_B,
                            DIM_M_B>(s_a1, s_b1, a1, b1, idx_a, idy_a, idx_b, idy_b);
            __syncthreads();
        }

        for(int k1 = 0; k1 < BLK_K; k1 += k_add)
        {
            shared_to_local<THR_N, THR_M, BLK_N, BLK_M, BLK_K, DIM_N, DIM_M>(
                a, b, s_a1, s_b1, k1, idx, idy);
            if constexpr(MINPLUS)
                compute_minplus<THR_N, THR_M>(a, b, c);
            else
                compute_plusmin<THR_N, THR_M>(a, b, c);
        }

        local_to_global<THR_N, THR_M, DIM_N, DIM_M, MINPLUS, BOUNDS>(
            d_c, d_d, ldc, ldd, beta, ci, cj, idx, idy, c, M, N);
    }

    template <int DIM_X, int DIM_Y, typename T, typename TScal, typename TConstPtr, typename TPtr>
    ROCBLAS_KERNEL(DIM_X* DIM_Y)
    geam_ex_scale_kernel(rocblas_int    m,
                         rocblas_int    n,
                         TScal          beta_host_device,
                         TConstPtr      dC,
                         rocblas_stride offset_c,
                         rocblas_int    ldc,
                         rocblas_stride stride_c,
                         TPtr           dD,
                         rocblas_stride offset_d,
                         rocblas_int    ldd,
                         rocblas_stride stride_d)
    {
        auto beta = load_scalar(beta_host_device);
        auto C    = beta ? load_ptr_batch(dC, blockIdx.z, offset_c, stride_c) : nullptr;
        auto D    = load_ptr_batch(dD, blockIdx.z, offset_d, stride_d);

        int num_blocksx = (m - 1) / DIM_X + 1;
        int blx         = blockIdx.x % num_blocksx;
        int bly         = blockIdx.x / num_blocksx;

        auto tx = blx * DIM_X + threadIdx.x;
        auto ty = bly * DIM_Y + threadIdx.y;

        if(tx < m && ty < n)
        {
            D[ty * size_t(ldd) + tx] = beta ? beta * C[ty * size_t(ldc) + tx] : 0;
        }
    }

    template <int DIM_X, int DIM_Y, typename T, typename TScal, typename TConstPtr, typename TPtr>
    ROCBLAS_KERNEL(DIM_X* DIM_Y)
    geam_ex_round_kernel(rocblas_int    m,
                         rocblas_int    n,
                         TScal          beta_host_device,
                         TConstPtr      dC,
                         rocblas_stride offset_c,
                         rocblas_int    ldc,
                         rocblas_stride stride_c,
                         TPtr           dD,
                         rocblas_stride offset_d,
                         rocblas_int    ldd,
                         rocblas_stride stride_d)
    {
        auto beta = load_scalar(beta_host_device);
        auto C    = beta ? load_ptr_batch(dC, blockIdx.z, offset_c, stride_c) : nullptr;
        auto D    = load_ptr_batch(dD, blockIdx.z, offset_d, stride_d);

        int num_blocksx = (m - 1) / DIM_X + 1;
        int blx         = blockIdx.x % num_blocksx;
        int bly         = blockIdx.x / num_blocksx;

        auto tx = blx * DIM_X + threadIdx.x;
        auto ty = bly * DIM_Y + threadIdx.y;

        if(tx < m && ty < n)
        {
            auto orig_val = beta ? beta * C[ty * size_t(ldc) + tx] : 0;
            if(orig_val > 0)
                D[ty * size_t(ldd) + tx] = 0;
            else
                D[ty * size_t(ldd) + tx] = orig_val;
        }
    }

    template <bool BATCHED, typename T, typename TConstPtr, typename TPtr>
    rocblas_status geam_ex_source_solution(rocblas_handle            handle,
                                           rocblas_operation         trans_a,
                                           rocblas_operation         trans_b,
                                           rocblas_int               m,
                                           rocblas_int               n,
                                           rocblas_int               k,
                                           const T*                  alpha,
                                           TConstPtr*                dA,
                                           rocblas_stride            offset_a,
                                           rocblas_int               lda,
                                           rocblas_stride            stride_a,
                                           TConstPtr*                dB,
                                           rocblas_stride            offset_b,
                                           rocblas_int               ldb,
                                           rocblas_stride            stride_b,
                                           const T*                  beta,
                                           TConstPtr*                dC,
                                           rocblas_stride            offset_c,
                                           rocblas_int               ldc,
                                           rocblas_stride            stride_c,
                                           TPtr*                     dD,
                                           rocblas_stride            offset_d,
                                           rocblas_int               ldd,
                                           rocblas_stride            stride_d,
                                           rocblas_int               batch_count,
                                           rocblas_geam_ex_operation geam_ex_op)
    {
        auto           stream   = handle->get_stream();
        auto           ptr_mode = handle->pointer_mode;
        TConstPtr*     dA_krn;
        TConstPtr*     dB_krn;
        TConstPtr*     dC_krn;
        TPtr*          dD_krn;
        rocblas_stride a_st_or_of;
        rocblas_stride b_st_or_of;
        rocblas_stride c_st_or_of;
        rocblas_stride d_st_or_of;

        if(BATCHED)
        {
            dA_krn     = dA;
            dB_krn     = dB;
            dC_krn     = dC;
            dD_krn     = dD;
            a_st_or_of = offset_a;
            b_st_or_of = offset_b;
            c_st_or_of = offset_c;
            d_st_or_of = offset_d;
        }
        else
        {
            dA_krn     = dA + offset_a;
            dB_krn     = dB + offset_b;
            dC_krn     = dC + offset_c;
            dD_krn     = dD + offset_d;
            a_st_or_of = stride_a;
            b_st_or_of = stride_b;
            c_st_or_of = stride_c;
            d_st_or_of = stride_d;
        }

        if(k == 0
           || (ptr_mode == rocblas_pointer_mode_host && *alpha == 0
               && geam_ex_op == rocblas_geam_ex_operation_plus_min))
        {
            static constexpr int GEAM_SCALE_DIM_X = 32;
            static constexpr int GEAM_SCALE_DIM_Y = 32;

            rocblas_int blocksX = (m - 1) / GEAM_SCALE_DIM_X + 1;
            rocblas_int blocksY = (n - 1) / GEAM_SCALE_DIM_Y + 1;
            blocksX *= blocksY; // overflow only on TB+

            dim3 geam_scale_grid(blocksX, 1, batch_count);
            dim3 geam_scale_threads(GEAM_SCALE_DIM_X, GEAM_SCALE_DIM_Y);

            if(ptr_mode == rocblas_pointer_mode_host)
                ROCBLAS_LAUNCH_KERNEL((geam_ex_scale_kernel<GEAM_SCALE_DIM_X, GEAM_SCALE_DIM_Y, T>),
                                      geam_scale_grid,
                                      geam_scale_threads,
                                      0,
                                      stream,
                                      m,
                                      n,
                                      *beta,
                                      dC,
                                      offset_c,
                                      ldc,
                                      stride_c,
                                      dD,
                                      offset_d,
                                      ldd,
                                      stride_d);
            else
                ROCBLAS_LAUNCH_KERNEL((geam_ex_scale_kernel<GEAM_SCALE_DIM_X, GEAM_SCALE_DIM_Y, T>),
                                      geam_scale_grid,
                                      geam_scale_threads,
                                      0,
                                      stream,
                                      m,
                                      n,
                                      beta,
                                      dC,
                                      offset_c,
                                      ldc,
                                      stride_c,
                                      dD,
                                      offset_d,
                                      ldd,
                                      stride_d);

            return rocblas_status_success;
        }

        if(ptr_mode == rocblas_pointer_mode_host && *alpha == 0)
        {
            static constexpr int GEAM_ROUND_DIM_X = 32;
            static constexpr int GEAM_ROUND_DIM_Y = 32;

            rocblas_int blocksX = (m - 1) / GEAM_ROUND_DIM_X + 1;
            rocblas_int blocksY = (n - 1) / GEAM_ROUND_DIM_Y + 1;
            blocksX *= blocksY; // overflow only on TB+

            dim3 geam_round_grid(blocksX, 1, batch_count);
            dim3 geam_round_threads(GEAM_ROUND_DIM_X, GEAM_ROUND_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((geam_ex_round_kernel<GEAM_ROUND_DIM_X, GEAM_ROUND_DIM_Y, T>),
                                  geam_round_grid,
                                  geam_round_threads,
                                  0,
                                  stream,
                                  m,
                                  n,
                                  *beta,
                                  dC,
                                  offset_c,
                                  ldc,
                                  stride_c,
                                  dD,
                                  offset_d,
                                  ldd,
                                  stride_d);

            return rocblas_status_success;
        }

#define LAUNCH_GEAM_SOURCE_KERNEL(                                      \
    TRANSA_, TRANSB_, DIM_M_A_, DIM_N_A_, DIM_M_B_, DIM_N_B_, MINPLUS_) \
    if(m % BLK_M == 0 && n % BLK_N == 0 && k % BLK_K == 0)              \
    {                                                                   \
        dim3 dimBlock(DIM_M, DIM_N, 1);                                 \
        dim3 dimGrid((m / BLK_M) * (n / BLK_N), 1, batch_count);        \
        if(ptr_mode == rocblas_pointer_mode_device)                     \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        false,          \
                                                        false,          \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  alpha,                                \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  beta,                                 \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
        else if(*alpha == 1)                                            \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        true,           \
                                                        false,          \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  *alpha,                               \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  *beta,                                \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
        else                                                            \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        false,          \
                                                        false,          \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  *alpha,                               \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  *beta,                                \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        dim3        dimBlock(DIM_M, DIM_N, 1);                          \
        rocblas_int blocksX = (m - 1) / BLK_M + 1;                      \
        rocblas_int blocksY = (n - 1) / BLK_N + 1;                      \
        blocksX *= blocksY; /* overflow only on TB+ */                  \
        dim3 dimGrid(blocksX, 1, batch_count);                          \
        if(ptr_mode == rocblas_pointer_mode_device)                     \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        false,          \
                                                        true,           \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  alpha,                                \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  beta,                                 \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
        else if(*alpha == 1)                                            \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        true,           \
                                                        true,           \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  *alpha,                               \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  *beta,                                \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
        else                                                            \
        {                                                               \
            ROCBLAS_LAUNCH_KERNEL((geam_min_plus_kernel<T,              \
                                                        Tab,            \
                                                        Tc,             \
                                                        DIM_M,          \
                                                        DIM_N,          \
                                                        BLK_M,          \
                                                        BLK_N,          \
                                                        BLK_K,          \
                                                        DIM_M_A_,       \
                                                        DIM_N_A_,       \
                                                        DIM_M_B_,       \
                                                        DIM_N_B_,       \
                                                        TRANSA_,        \
                                                        TRANSB_,        \
                                                        false,          \
                                                        true,           \
                                                        MINPLUS_>),     \
                                  dimGrid,                              \
                                  dimBlock,                             \
                                  0,                                    \
                                  stream,                               \
                                  m,                                    \
                                  n,                                    \
                                  k,                                    \
                                  *alpha,                               \
                                  dA_krn,                               \
                                  lda,                                  \
                                  a_st_or_of,                           \
                                  dB_krn,                               \
                                  ldb,                                  \
                                  b_st_or_of,                           \
                                  *beta,                                \
                                  dC_krn,                               \
                                  ldc,                                  \
                                  c_st_or_of,                           \
                                  dD_krn,                               \
                                  ldd,                                  \
                                  d_st_or_of,                           \
                                  batch_count,                          \
                                  geam_ex_op);                          \
        }                                                               \
    }

        constexpr rocblas_int DIM_M_A = 64;
        constexpr rocblas_int DIM_N_A = 4;
        constexpr rocblas_int DIM_M_B = 4;
        constexpr rocblas_int DIM_N_B = 64;
        if(geam_ex_op == rocblas_geam_ex_operation_min_plus)
        {
            if(trans_a == rocblas_operation_none && trans_b == rocblas_operation_none)
            {
                // NN
                constexpr rocblas_int BLK_M = 256;
                constexpr rocblas_int BLK_N = 64;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 32;
                constexpr rocblas_int DIM_N = 8;
                if constexpr(std::is_same_v<rocblas_half, T>)
                {
                    using Tc  = array2_t<T>;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, true);
                }
                else if constexpr(std::is_same_v<float, T>)
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, true);
                }
                else
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, true);
                }
            }
            else if(trans_a != rocblas_operation_none && trans_b == rocblas_operation_none)
            {
                // TN
                constexpr rocblas_int BLK_M = 128;
                constexpr rocblas_int BLK_N = 128;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 32;
                constexpr rocblas_int DIM_N = 8;
                if constexpr(std::is_same_v<rocblas_half, T>)
                {
                    using Tc  = array2_t<T>;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, true);
                }
                else if constexpr(std::is_same_v<float, T>)
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, true);
                }
                else
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, true);
                }
            }
            else if(trans_a == rocblas_operation_none && trans_b != rocblas_operation_none)
            {
                // NT
                constexpr rocblas_int BLK_M = 64;
                constexpr rocblas_int BLK_N = 256;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 8;
                constexpr rocblas_int DIM_N = 32;
                if constexpr(std::is_same_v<rocblas_half, T>)
                {
                    using Tc  = array2_t<T>;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, true);
                }
                else if constexpr(std::is_same_v<float, T>)
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, true);
                }
                else
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, true);
                }
            }
            else
            {
                // TT
                constexpr rocblas_int BLK_M = 64;
                constexpr rocblas_int BLK_N = 256;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 8;
                constexpr rocblas_int DIM_N = 32;
                if constexpr(std::is_same_v<rocblas_half, T>)
                {
                    using Tc  = array2_t<T>;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, true);
                }
                else if constexpr(std::is_same_v<float, T>)
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, true);
                }
                else
                {
                    using Tc  = T;
                    using Tab = array2_t<T>;
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, true);
                }
            }
        }
        else if(geam_ex_op == rocblas_geam_ex_operation_plus_min)
        {
            if constexpr(std::is_same_v<rocblas_half, T>)
            {
                using Tc                    = array2_t<T>;
                using Tab                   = array2_t<T>;
                constexpr rocblas_int BLK_M = 64;
                constexpr rocblas_int BLK_N = 128;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 8;
                constexpr rocblas_int DIM_N = 32;
                if(trans_a == rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // NN
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a != rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // TN
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a == rocblas_operation_none && trans_b != rocblas_operation_none)
                {
                    // NT
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, false);
                }
                else
                {
                    // TT
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, false);
                }
            }
            else if constexpr(std::is_same_v<float, T>)
            {
                using Tc                    = array2_t<T>;
                using Tab                   = array2_t<T>;
                constexpr rocblas_int BLK_M = 64;
                constexpr rocblas_int BLK_N = 128;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 8;
                constexpr rocblas_int DIM_N = 32;
                if(trans_a == rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // NN
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a != rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // TN
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a == rocblas_operation_none && trans_b != rocblas_operation_none)
                {
                    // NT
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, false);
                }
                else
                {
                    // TT
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, false);
                }
            }
            else if(std::is_same_v<double, T>)
            {
                using Tc                    = T;
                using Tab                   = T;
                constexpr rocblas_int BLK_M = 128;
                constexpr rocblas_int BLK_N = 128;
                constexpr rocblas_int BLK_K = 4;
                constexpr rocblas_int DIM_M = 4;
                constexpr rocblas_int DIM_N = 64;
                if(trans_a == rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // NN
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'N', DIM_M_A, DIM_N_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a != rocblas_operation_none && trans_b == rocblas_operation_none)
                {
                    // TN
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'N', DIM_N_A, DIM_M_A, DIM_M_B, DIM_N_B, false);
                }
                else if(trans_a == rocblas_operation_none && trans_b != rocblas_operation_none)
                {
                    // NT
                    LAUNCH_GEAM_SOURCE_KERNEL('N', 'T', DIM_M_A, DIM_N_A, DIM_N_B, DIM_M_B, false);
                }
                else
                {
                    // TT
                    LAUNCH_GEAM_SOURCE_KERNEL('T', 'T', DIM_N_A, DIM_M_A, DIM_N_B, DIM_M_B, false);
                }
            }
        }
#undef LAUNCH_GEAM_SOURCE_KERNEL

        return rocblas_status_success;
    }

}

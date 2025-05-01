/************************************************************************
 * Derived from
 * Gotlub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
 * and
 * Hari & Kovac (2019). On the Convergence of Complex Jacobi Methods.
 *     Linear and Multilinear Algebra 69(3), p. 489-514.
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_syev_heev.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/************** Kernels and device functions for small size*******************/
/*****************************************************************************/

#define SYEVJ_BDIM 1024 // Max number of threads per thread-block used in syevj_small kernel

/** SYEVJ_SMALL_KERNEL/RUN_SYEVJ applies the Jacobi eigenvalue algorithm to matrices of size
    n <= SYEVJ_BLOCKED_SWITCH. For each off-diagonal element A[i,j], a Jacobi rotation J is
    calculated so that (J'AJ)[i,j] = 0. J only affects rows i and j, and J' only affects
    columns i and j. Therefore, ceil(n / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict between threads. We use top/bottom pairs
    to obtain i's and j's that do not conflict, and cycle them to cover all off-diagonal indices.

    (Call the syevj_small_kernel with batch_count groups in z, of dim = ddx * ddy threads in x.
	Then, the run_syevj device function will be run by all threads organized in a ddx-by-ddy array.
	Normally, ddx <= ceil(n / 2), and ddy <= ceil(n / 2). Any thread with index i >= ceil(n / 2) or
	j >= ceil(n / 2) will not execute any computations). **/
template <typename T, typename S>
__device__ void run_syevj(const rocblas_int dimx,
                          const rocblas_int dimy,
                          const rocblas_int tix,
                          const rocblas_int tiy,
                          const rocblas_esort esort,
                          const rocblas_evect evect,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          T* A,
                          const rocblas_int lda,
                          const S abstol,
                          const S eps,
                          S* residual,
                          const rocblas_int max_sweeps,
                          rocblas_int* n_sweeps,
                          S* W,
                          rocblas_int* info,
                          T* Acpy,
                          S* cosines_res,
                          T* sines_diag,
                          rocblas_int* top,
                          rocblas_int* bottom)
{
    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    S local_res = 0;
    S local_diag = 0;

    if(tiy == 0)
    {
        // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
        // squared Frobenius norm (first by column/row then sum)
        if(uplo == rocblas_fill_upper)
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = n - 1; j > i; j--)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
        }
        else
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = 0; j < i; j++)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
        }
        cosines_res[tix] = local_res;
        sines_diag[tix] = local_diag;

        // initialize top/bottom pairs
        for(i = tix; i < half_n; i += dimx)
        {
            top[i] = i * 2;
            bottom[i] = i * 2 + 1;
        }
    }
    __syncthreads();

    // set tolerance
    local_res = 0;
    local_diag = 0;
    for(i = 0; i < dimx; i++)
    {
        local_res += cosines_res[i];
        local_diag += std::real(sines_diag[i]);
    }
    S tolerance = (local_res + local_diag) * abstol * abstol;
    S small_num = get_safemin<S>() / eps;

    // execute sweeps
    rocblas_int count = (half_n - 1) / dimx + 1;
    while(sweeps < max_sweeps && local_res > tolerance)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        for(rocblas_int k = 0; k < even_n - 1; ++k)
        {
            for(rocblas_int cc = 0; cc < count; ++cc)
            {
                // get current top/bottom pair
                rocblas_int kx = tix + cc * dimx;
                i = kx < half_n ? top[kx] : n;
                j = kx < half_n ? bottom[kx] : n;

                // calculate current rotation J
                if(tiy == 0 && i < n && j < n)
                {
                    aij = Acpy[i + j * n];
                    mag = std::abs(aij);

                    if(mag * mag < small_num)
                    {
                        c = 1;
                        s1 = 0;
                    }
                    else
                    {
                        g = 2 * mag;
                        f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
                        f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                        lartg(f, g, c, s, r);
                        s1 = s * aij / mag;
                    }
                    cosines_res[tix] = c;
                    sines_diag[tix] = s1;
                }
                __syncthreads();

                // apply J from the right and update vectors
                if(i < n && j < n)
                {
                    c = cosines_res[tix];
                    s1 = sines_diag[tix];
                    s2 = conj(s1);

                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[y1 + i * n];
                        temp2 = Acpy[y1 + j * n];
                        Acpy[y1 + i * n] = c * temp1 + s2 * temp2;
                        Acpy[y1 + j * n] = -s1 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[y2 + i * n];
                            temp2 = Acpy[y2 + j * n];
                            Acpy[y2 + i * n] = c * temp1 + s2 * temp2;
                            Acpy[y2 + j * n] = -s1 * temp1 + c * temp2;
                        }

                        if(evect != rocblas_evect_none)
                        {
                            temp1 = A[y1 + i * lda];
                            temp2 = A[y1 + j * lda];
                            A[y1 + i * lda] = c * temp1 + s2 * temp2;
                            A[y1 + j * lda] = -s1 * temp1 + c * temp2;
                            if(y2 < n)
                            {
                                temp1 = A[y2 + i * lda];
                                temp2 = A[y2 + j * lda];
                                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
                            }
                        }
                    }
                }
                __syncthreads();

                // apply J' from the left
                if(i < n && j < n)
                {
                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[i + y1 * n];
                        temp2 = Acpy[j + y1 * n];
                        Acpy[i + y1 * n] = c * temp1 + s1 * temp2;
                        Acpy[j + y1 * n] = -s2 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[i + y2 * n];
                            temp2 = Acpy[j + y2 * n];
                            Acpy[i + y2 * n] = c * temp1 + s1 * temp2;
                            Acpy[j + y2 * n] = -s2 * temp1 + c * temp2;
                        }
                    }
                }
                __syncthreads();

                // round aij and aji to zero
                if(tiy == 0 && i < n && j < n)
                {
                    Acpy[i + j * n] = 0;
                    Acpy[j + i * n] = 0;
                }
                __syncthreads();

                // rotate top/bottom pair
                if(tiy == 0 && kx < half_n)
                {
                    if(i > 0)
                    {
                        if(i == 2 || i == even_n - 1)
                            top[kx] = i - 1;
                        else
                            top[kx] = i + ((i % 2 == 0) ? -2 : 2);
                    }
                    if(j == 2 || j == even_n - 1)
                        bottom[kx] = j - 1;
                    else
                        bottom[kx] = j + ((j % 2 == 0) ? -2 : 2);
                }
                __syncthreads();
            }
        }

        // update norm
        if(tiy == 0)
        {
            local_res = 0;

            for(i = tix; i < n; i += dimx)
            {
                for(j = 0; j < i; j++)
                    local_res += 2 * std::norm(Acpy[i + j * n]);
            }
            cosines_res[tix] = local_res;
        }
        __syncthreads();

        local_res = 0;
        for(i = 0; i < dimx; i++)
            local_res += cosines_res[i];

        sweeps++;
    }

    // finalize outputs
    if(tiy == 0)
    {
        if(tix == 0)
        {
            *residual = sqrt(local_res);
            if(sweeps <= max_sweeps)
            {
                *n_sweeps = sweeps;
                *info = 0;
            }
            else
            {
                *n_sweeps = max_sweeps;
                *info = 1;
            }
        }

        // update W
        for(i = tix; i < n; i += dimx)
            W[i] = std::real(Acpy[i + i * n]);
    }
    __syncthreads();

    // if no sort, then stop
    if(esort == rocblas_esort_none)
        return;

    //otherwise sort eigenvalues and eigenvectors by selection sort
    rocblas_int m;
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j && tiy == 0)
        {
            if(tix == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tix; i < n; i += dimx)
                    swap(A[i + m * lda], A[i + j * lda]);
            }
        }
        __syncthreads();
    }
}

__host__ __device__ inline void
    syevj_get_dims(rocblas_int n, rocblas_int bdim, rocblas_int* ddx, rocblas_int* ddy)
{
    // (TODO: Some tuning could be beneficial in the future.
    //	For now, we use a max of BDIM = ddx * ddy threads.
    //	ddy is set to min(BDIM/4, ceil(n/2)) and ddx to min(BDIM/ddy, ceil(n/2)).

    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    rocblas_int y = std::min(bdim / 4, half_n);
    rocblas_int x = std::min(bdim / y, half_n);
    *ddx = x;
    *ddy = y;
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYEVJ_BDIM) syevj_small_kernel(const rocblas_esort esort,
                                                                       const rocblas_evect evect,
                                                                       const rocblas_fill uplo,
                                                                       const rocblas_int n,
                                                                       U AA,
                                                                       const rocblas_int shiftA,
                                                                       const rocblas_int lda,
                                                                       const rocblas_stride strideA,
                                                                       const S abstol,
                                                                       const S eps,
                                                                       S* residualA,
                                                                       const rocblas_int max_sweeps,
                                                                       rocblas_int* n_sweepsA,
                                                                       S* WW,
                                                                       const rocblas_stride strideW,
                                                                       rocblas_int* infoA,
                                                                       T* AcpyA)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;
    S* W = WW + bid * strideW;
    S* residual = residualA + bid;
    rocblas_int* n_sweeps = n_sweepsA + bid;
    rocblas_int* info = infoA + bid;

    // get dimensions of 2D thread array
    rocblas_int ddx, ddy;
    syevj_get_dims(n, SYEVJ_BDIM, &ddx, &ddy);

    // shared memory
    extern __shared__ double lmem[];
    S* cosines_res = reinterpret_cast<S*>(lmem);
    T* sines_diag = reinterpret_cast<T*>(cosines_res + ddx);
    rocblas_int* top = reinterpret_cast<rocblas_int*>(sines_diag + ddx);
    rocblas_int* bottom = top + half_n;

    // re-arrange threads in 2D array
    rocblas_int tix = tid / ddy;
    rocblas_int tiy = tid % ddy;

    // execute
    run_syevj(ddx, ddy, tix, tiy, esort, evect, uplo, n, A, lda, abstol, eps, residual, max_sweeps,
              n_sweeps, W, info, Acpy, cosines_res, sines_diag, top, bottom);
}

/************** Kernels and device functions for large size*******************/
/*****************************************************************************/

/** SYEVJ_INIT copies A to Acpy, calculates the residual norm of the matrix, and
    initializes the top/bottom pairs.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_init(const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int half_blocks,
                                 const rocblas_int n,
                                 U AA,
                                 const rocblas_int shiftA,
                                 const rocblas_int lda,
                                 const rocblas_stride strideA,
                                 S abstol,
                                 S* residual,
                                 T* AcpyA,
                                 S* norms,
                                 rocblas_int* top,
                                 rocblas_int* bottom,
                                 rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int dimx = hipBlockDim_x;

    // local variables
    T temp;
    rocblas_int i, j;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);
    S* sh_diag = sh_res + dimx;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (by column/row)
    S local_res = 0;
    S local_diag = 0;
    if(uplo == rocblas_fill_upper)
    {
        for(i = tid; i < n; i += dimx)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = n - 1; j > i; j--)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    else
    {
        for(i = tid; i < n; i += dimx)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = 0; j < i; j++)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    sh_res[tid] = local_res;
    sh_diag[tid] = local_diag;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < std::min(n, dimx); i++)
        {
            local_res += sh_res[i];
            local_diag += sh_diag[i];
        }

        norms[bid] = (local_res + local_diag) * abstol * abstol;
        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = 1;
            atomicAdd(completed, 1);
        }
    }

    // initialize top/bottom pairs
    if(bid == 0 && top && bottom)
    {
        for(i = tid; i < half_blocks; i += dimx)
        {
            top[i] = 2 * i;
            bottom[i] = 2 * i + 1;
        }
    }
}

/** SYEVJ_DIAG_KERNEL decomposes diagonal blocks of size nb <= BS2. For each off-diagonal element
    A[i,j], a Jacobi rotation J is calculated so that (J'AJ)[i,j] = 0. J only affects rows i and j,
    and J' only affects columns i and j. Therefore, ceil(nb / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict between threads. We use top/bottom pairs
    to obtain i's and j's that do not conflict, and cycle them to cover all off-diagonal indices.

    Call this kernel with batch_count blocks in z, and BS2 / 2 threads in x and y. Each thread block
    will work on a separate diagonal block; for a matrix consisting of b * b blocks, use b thread
    blocks in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_diag_kernel(const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const S eps,
                                        T* JA,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int nb_max = 2 * hipBlockDim_x;
    rocblas_int offset = hipBlockIdx_x * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j, k;
    rocblas_int xx1 = 2 * tix, xx2 = xx1 + 1;
    rocblas_int yy1 = 2 * tiy, yy2 = yy1 + 1;
    rocblas_int x1 = xx1 + offset, x2 = x1 + 1;
    rocblas_int y1 = yy1 + offset, y2 = y1 + 1;

    rocblas_int half_n = (n - 1) / 2 + 1;
    rocblas_int nb = std::min(2 * half_n - offset, nb_max);
    rocblas_int half_nb = nb / 2;

    if(tix >= half_nb || tiy >= half_nb)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + half_nb);
    rocblas_int* sh_top = reinterpret_cast<rocblas_int*>(sh_sines + half_nb);
    rocblas_int* sh_bottom = sh_top + half_nb;

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * nb_max] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * nb_max] = 0;
        J[xx2 + yy1 * nb_max] = 0;
        J[xx2 + yy2 * nb_max] = (xx2 == yy2 ? 1 : 0);
    }

    // initialize top/bottom
    if(tiy == 0)
    {
        sh_top[tix] = x1;
        sh_bottom[tix] = x2;
    }

    S small_num = get_safemin<S>() / eps;

    // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to A
    i = x1;
    j = x2;
    for(k = 0; k < nb - 1; k++)
    {
        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag * mag < small_num)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offset;
                xx2 = j - offset;
                temp1 = J[xx1 + yy1 * nb_max];
                temp2 = J[xx2 + yy1 * nb_max];
                J[xx1 + yy1 * nb_max] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * nb_max] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * nb_max];
                    temp2 = J[xx2 + yy2 * nb_max];
                    J[xx1 + yy2 * nb_max] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * nb_max] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && i < n && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }

        // cycle top/bottom pairs
        if(tix == 1)
            i = sh_bottom[0];
        else if(tix > 1)
            i = sh_top[tix - 1];
        if(tix == half_nb - 1)
            j = sh_top[half_nb - 1];
        else
            j = sh_bottom[tix + 1];
        __syncthreads();

        if(tiy == 0)
        {
            sh_top[tix] = i;
            sh_bottom[tix] = j;
        }
    }
}

/** SYEVJ_DIAG_ROTATE rotates off-diagonal blocks of size nb <= BS2 using the rotations calculated
    by SYEVJ_DIAG_KERNEL.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on a separate off-diagonal block; for a matrix consisting of b * b blocks, use b groups
    in x and b - 1 groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_diag_rotate_org(const bool skip_block,
                                            const rocblas_int n,
                                            U AA,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            T* JA,
                                            rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + bix;

    if(completed[bid + 1])
        return;
    if(skip_block && bix == biy)
        return;

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offsetx = bix * nb_max;
    rocblas_int offsety = biy * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = std::min(n - offsetx, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += J[tix + k * nb_max] * A[y + (k + offsetx) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + k * nb_max]) * A[(k + offsetx) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}

template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_diag_rotate(const bool skip_block,
                                        const rocblas_int nb_max,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* JA,
                                        rocblas_int* completed,
                                        rocblas_int batch_count)
{
    bool constexpr APPLY_RIGHT = (!APPLY_LEFT);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;

    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    auto const bix_start = hipBlockIdx_x;
    auto const bix_inc = hipGridDim_x;

    auto const biy_start = hipBlockIdx_y;
    auto const biy_inc = hipGridDim_y;

    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const blocks = ceil(n, nb_max);

    // -------------------------------------------------
    // function to calculation the size of the i-th block
    // -------------------------------------------------
    auto bsize = [=](auto iblock) {
        auto const nb_last = n - (blocks - 1) * nb_max;
        bool const is_last_block = (iblock == (blocks - 1));
        return ((is_last_block) ? nb_last : nb_max);
    };

    extern double lmem[];
    size_t total_bytes = 0;
    T* pfree = reinterpret_cast<T*>(&(lmem[0]));

    auto const len_Ash = nb_max * nb_max;

    T* Ash_ = pfree;
    pfree += len_Ash;
    total_bytes += sizeof(T) * len_Ash;

    auto Ash = [=](auto i, auto j) -> T& {
        auto const ldAsh = nb_max;
        return (Ash_[i + j * ldAsh]);
    };

    // ----------------------------------------------------------------
    // Need to store a copy of A[iblock,kblock] or A[kblock,iblock] in LDS
    // to implement in-place update
    // ----------------------------------------------------------------
    size_t const max_lds = 64 * 1024;
    assert(total_bytes <= max_lds);

    size_t const len_Jsh = nb_max * nb_max;
    T* Jsh_ = pfree;
    pfree += len_Jsh;
    total_bytes += sizeof(T) * len_Jsh;

    // ----------------------------------------------------------------
    // Optional to store J into LDS if there is sufficient space in LDS
    // ----------------------------------------------------------------
    bool const use_Jsh = (total_bytes <= max_lds);

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        if(completed[bid + 1])
            continue;

        T* const A_ = load_ptr_batch<T>(AA, bid, shiftA, strideA);

        auto A = [=](auto i, auto j) -> T& { return (A_[i + j * static_cast<int64_t>(lda)]); };

        for(auto bix = bix_start; bix < blocks; bix += bix_inc)
        {
            auto const jid = bix + bid * blocks;

            T* const J_ = JA + (jid * (nb_max * nb_max));

            T* Jmat_ = (use_Jsh) ? Jsh_ : J_;
            auto Jmat = [=](auto i, auto j) -> const T {
                auto const ldj = nb_max;
                return (Jmat_[i + j * ldj]);
            };

            auto const iblock = bix;
            auto const nrowsJ = bsize(iblock);
            auto const ncolsJ = nrowsJ;

            if(use_Jsh)
            {
                // ---------------
                // load J into Jsh in LDS
                // ---------------
                __syncthreads();
                auto const ij_start = i_start + j_start * i_inc;
                auto const ij_inc = i_inc * j_inc;
                auto const len_Jsh = nb_max * nb_max;

                for(auto ij = ij_start; ij < len_Jsh; ij += ij_inc)
                {
                    Jsh_[ij] = J_[ij];
                }
                __syncthreads();
            }

            for(auto biy = biy_start; biy < blocks; biy += biy_inc)
            {
                auto const kblock = biy;
                bool const is_diagonal_block = (iblock == kblock);

                // -----------------------------------------------------------
                // need to skip the diagonal when updating matrix A
                // since the diagonal block has already been updated when
                // generating the rotation matrix J
                // However, diagonal block must also be updated when processing
                // matrix V of eigenvectors
                // -----------------------------------------------------------
                if(skip_block && is_diagonal_block)
                    continue;

                auto const offseti = iblock * nb_max;
                auto const offsetk = kblock * nb_max;

                //  -------------------------------------
                //  use A[iblock,kblock] if (APPLY_LEFT)
                //  use A[kblock,iblock] if (APPLY_RIGHT)
                //  -------------------------------------
                auto const nrows_Ash = (APPLY_LEFT) ? bsize(iblock) : bsize(kblock);
                auto const ncols_Ash = (APPLY_LEFT) ? bsize(kblock) : bsize(iblock);

                // ----------------------------------------------
                // functions to map between local index (tix,tiy)
                // to global index (ia,ja)
                // ----------------------------------------------
                auto get_ia = [=](auto tix) {
                    auto const ia = (APPLY_LEFT) ? tix + offseti : tix + offsetk;
                    return (ia);
                };

                auto get_ja = [=](auto tiy) {
                    auto const ja = (APPLY_LEFT) ? tiy + offsetk : tiy + offseti;
                    return (ja);
                };

                {
                    //  ------------------------------------------------
                    //  load block of A[iblock,kblock]  into Ash in LDS if (APPLY_LEFT)
                    //  load block of A[kblock,iblock]  into Ash in LDS if (APPLY_RIGHT)
                    //  ------------------------------------------------
                    __syncthreads();

                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            auto const ia = get_ia(i);
                            auto const ja = get_ja(j);
                            Ash(i, j) = A(ia, ja);
                        }
                    }
                    __syncthreads();
                }

                // ----------------------------
                // apply J to the current block
                // ----------------------------
                if(APPLY_RIGHT)
                {
                    // ---------------------
                    // A <- A * transpose(J)
                    // ---------------------
                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            T aij = 0;
                            for(auto k = 0; k < ncolsJ; k++)
                            {
                                auto const aik = Ash(i, k);
                                auto const Jt_kj = Jmat(j, k);

                                aij += aik * Jt_kj;
                            }
                            auto const ia = get_ia(i);
                            auto const ja = get_ja(j);
                            A(ia, ja) = aij;
                        }
                    }

                    __syncthreads();
                }
                else
                {
                    // ----------------
                    // APPLY_LEFT
                    // A <- conj(J) * A
                    // ----------------

                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            T aij = 0;
                            for(auto k = 0; k < ncolsJ; k++)
                            {
                                auto const Jc_ik = conj(Jmat(i, k));
                                auto const akj = Ash(k, j);
                                aij += Jc_ik * akj;
                            }
                            auto const ia = get_ia(i);
                            auto const ja = get_ja(j);
                            A(ia, ja) = aij;
                        }
                    }
                    __syncthreads();
                }

            } // end for kblock
        } // end for iblock
    } // end for bid
}

/** SYEVJ_OFFD_KERNEL decomposes off-diagonal blocks of size nb <= BS2. For each element in the block
    (which is an off-diagonal element A[i,j] in the matrix A), a Jacobi rotation J is calculated so that
    (J'AJ)[i,j] = 0. J only affects rows i and j, and J' only affects columns i and j. Therefore,
    nb rotations can be computed and applied in parallel, so long as the rotations do not conflict between
    threads. We select the initial set of i's and j's to span the block's diagonal, and iteratively move
    to the right (wrapping around as necessary) to cover all indices.

    Since A[i,i], A[j,j], and A[j,i] are all in separate blocks, we also need to ensure that
    rotations do not conflict between thread groups. We use block-level top/bottom pairs
    to obtain off-diagonal block indices that do not conflict.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on four matrix blocks; for a matrix consisting of b * b blocks, use b / 2 groups in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_kernel_org(const rocblas_int blocks,
                                            const rocblas_int n,
                                            U AA,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            const S eps,
                                            T* JA,
                                            rocblas_int* top,
                                            rocblas_int* bottom,
                                            rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[hipBlockIdx_x];
    rocblas_int j = bottom[hipBlockIdx_x];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int k;
    rocblas_int xx1 = tix, xx2 = tix + nb_max;
    rocblas_int yy1 = tiy, yy2 = tiy + nb_max;
    rocblas_int x1 = tix + offseti, x2 = tix + offsetj;
    rocblas_int y1 = tiy + offseti, y2 = tiy + offsetj;

    if(y1 >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * 4 * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + nb_max);

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * ldj] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * ldj] = 0;
        J[xx2 + yy1 * ldj] = 0;
        J[xx2 + yy2 * ldj] = (xx2 == yy2 ? 1 : 0);
    }

    S small_num = get_safemin<S>() / eps;

    // for each element, calculate the Jacobi rotation and apply it to A
    for(k = 0; k < nb_max; k++)
    {
        // get element indices
        i = x1;
        j = (tix + k) % nb_max + offsetj;

        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag * mag < small_num)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offseti;
                xx2 = j - offsetj + nb_max;
                temp1 = J[xx1 + yy1 * ldj];
                temp2 = J[xx2 + yy1 * ldj];
                J[xx1 + yy1 * ldj] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * ldj] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * ldj];
                    temp2 = J[xx2 + yy2 * ldj];
                    J[xx1 + yy2 * ldj] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * ldj] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }
    }
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_kernel(const rocblas_int nb_max,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const S eps,
                                        T* JA,
                                        rocblas_int* top,
                                        rocblas_int* bottom,
                                        rocblas_int* completed,
                                        rocblas_int batch_count)
{
    auto const blocks = ceil(n, nb_max);
    auto const even_blocks = blocks + (blocks % 2);
    auto const half_blocks = even_blocks / 2;

    auto const ibx = hipBlockIdx_x;
    auto const iby = hipBlockIdx_y;
    auto const ibz = hipBlockIdx_z;

    auto const nbx = hipGridDim_x;
    auto const nby = hipGridDim_y;
    auto const nbz = hipGridDim_z;

    auto const tix_start = hipThreadIdx_x;
    auto const tiy_start = hipThreadIdx_y;
    auto const tix_inc = hipBlockDim_x;
    auto const tiy_inc = hipBlockDim_y;

    auto const bid_start = ibz;
    auto const bid_inc = nbz;

    // shared memory
    extern __shared__ double lmem[];

    size_t total_bytes = 0;
    std::byte* pfree = reinterpret_cast<std::byte*>(&(lmem[0]));

    size_t const size_sh_cosines = sizeof(S) * nb_max;
    S* const sh_cosines = reinterpret_cast<S*>(pfree);
    pfree += size_sh_cosines;
    total_bytes += size_sh_cosines;

    size_t const size_sh_sines = sizeof(T) * nb_max;
    T* const sh_sines = reinterpret_cast<T*>(pfree);
    pfree += size_sh_sines;
    total_bytes += size_sh_sines;

    size_t const size_Jsh = sizeof(T) * (2 * nb_max) * (2 * nb_max);
    T* const Jsh_ = reinterpret_cast<T*>(pfree);
    pfree += size_Jsh;
    total_bytes += size_Jsh;

    size_t const max_lds = 64 * 1024;
    bool const use_Jsh = (total_bytes <= max_lds);

    auto bsize = [=](auto iblock) {
        auto const nb_last = n - (blocks - 1) * nb_max;
        bool const is_last_block = (iblock == (blocks - 1));
        return ((is_last_block) ? nb_last : nb_max);
    };

    auto const ipair_start = ibx;
    auto const ipair_inc = nbx;
    auto const npairs = half_blocks;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        if(completed[bid + 1])
            continue;

        T* const A_ = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        auto A = [=](auto i, auto j) -> T& { return (A_[i + j * static_cast<int64_t>(lda)]); };

        for(auto ipair = ipair_start; ipair < npairs; ipair += ipair_inc)
        {
            auto const iblock = std::min(top[ipair], bottom[ipair]);
            auto const jblock = std::max(top[ipair], bottom[ipair]);

            bool const is_valid_block = (iblock < blocks) && (jblock < blocks);
            if(!is_valid_block)
                continue;

            auto const offseti = iblock * nb_max;
            auto const offsetj = jblock * nb_max;

            auto const ni = bsize(iblock);
            auto const nj = bsize(jblock);
            auto const nrowsJ = ni + nj;
            auto const ncolsJ = nrowsJ;

            auto const jid = ipair + bid * npairs;
            T* const J_ = ((JA != nullptr) ? JA + (jid * (4 * nb_max * nb_max)) : nullptr);

            auto const ldj = (2 * nb_max);
            auto J = [=](auto i, auto j) -> T& { return (J_[i + j * ldj]); };

            T* const Jmat_ = (use_Jsh) ? Jsh_ : J_;
            auto Jmat = [=](auto i, auto j) -> T& { return (Jmat_[i + j * ldj]); };
            // ------------------------------------------
            // initialize Jmat to be the identity matrix
            // ------------------------------------------
            if(J_ != nullptr)
            {
                __syncthreads();

                auto const i_start = tix_start;
                auto const i_inc = tix_inc;

                auto const j_start = tiy_start;
                auto const j_inc = tiy_inc;

                for(auto j = j_start; j < ncolsJ; j += j_inc)
                {
                    for(auto i = i_start; i < nrowsJ; i += i_inc)
                    {
                        bool const is_diagonal = (i == j);
                        Jmat(i, j) = (is_diagonal) ? 1 : 0;
                    }
                }
                __syncthreads();
            }

            S const small_num = get_safemin<S>() / eps;
            S const sqrt_small_num = std::sqrt(small_num);

            // for each element, calculate the Jacobi rotation and apply it to A
            for(rocblas_int k = 0; k < nb_max; k++)
            {
                for(auto tiy = tiy_start; tiy < nb_max; tiy += tiy_inc)
                {
                    for(auto tix = tix_start; tix < nb_max; tix += tix_inc)
                    {
                        auto const x1 = tix + offseti;
                        auto const x2 = tix + offsetj;
                        auto const y1 = tiy + offseti;
                        auto const y2 = tiy + offsetj;

                        // get element indices
                        auto const i = x1;
                        auto const j = (tix + k) % nb_max + offsetj;

                        if((tiy == 0) && (i < n) && (j < n))
                        {
                            auto const aij = A(i, j);
                            auto const mag = std::abs(aij);

                            // identity rotation
                            S c = 1;
                            T s1 = 0;

                            // calculate rotation J
                            // bool const is_small_aij = (mag < sqrt_small_num);
                            bool const is_small_aij = (mag * mag < small_num);

                            if(!is_small_aij)
                            {
                                S g = 2 * mag;
                                S f = std::real(A(j, j) - A(i, i));
                                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);

                                S s = 0;
                                S r = 1;

                                lartg(f, g, c, s, r);
                                s1 = s * aij / mag;
                            }

                            sh_cosines[tix] = c;
                            sh_sines[tix] = s1;
                        }
                    }
                }
                __syncthreads();

                for(auto tiy = tiy_start; tiy < nb_max; tiy += tiy_inc)
                {
                    for(auto tix = tix_start; tix < nb_max; tix += tix_inc)
                    {
                        auto const x1 = tix + offseti;
                        auto const x2 = tix + offsetj;
                        auto const y1 = tiy + offseti;
                        auto const y2 = tiy + offsetj;

                        // get element indices
                        auto const i = x1;
                        auto const j = (tix + k) % nb_max + offsetj;

                        if((i < n) && (j < n))
                        {
                            auto const c = sh_cosines[tix];
                            auto const s1 = sh_sines[tix];
                            auto const s2 = conj(s1);

                            // store J row-wise
                            if(J_ != nullptr)
                            {
                                auto const xx1 = i - offseti;
                                auto const xx2 = j - offsetj + nb_max;
                                auto const yy1 = tiy;
                                auto const yy2 = tiy + nb_max;

                                {
                                    auto const temp1 = Jmat(xx1, yy1);
                                    auto const temp2 = Jmat(xx2, yy1);

                                    Jmat(xx1, yy1) = c * temp1 + s2 * temp2;
                                    Jmat(xx2, yy1) = -s1 * temp1 + c * temp2;
                                }

                                if(y2 < n)
                                {
                                    auto const temp1 = Jmat(xx1, yy2);
                                    auto const temp2 = Jmat(xx2, yy2);

                                    Jmat(xx1, yy2) = c * temp1 + s2 * temp2;
                                    Jmat(xx2, yy2) = -s1 * temp1 + c * temp2;
                                }
                            }

                            // apply J from the right
                            {
                                auto const temp1 = A(y1, i);
                                auto const temp2 = A(y1, j);
                                A(y1, i) = c * temp1 + s2 * temp2;
                                A(y1, j) = -s1 * temp1 + c * temp2;
                            }

                            if(y2 < n)
                            {
                                auto const temp1 = A(y2, i);
                                auto const temp2 = A(y2, j);
                                A(y2, i) = c * temp1 + s2 * temp2;
                                A(y2, j) = -s1 * temp1 + c * temp2;
                            }
                        }
                    }
                }

                __syncthreads();

                for(auto tiy = tiy_start; tiy < nb_max; tiy += tiy_inc)
                {
                    for(auto tix = tix_start; tix < nb_max; tix += tix_inc)
                    {
                        auto const x1 = tix + offseti;
                        auto const x2 = tix + offsetj;
                        auto const y1 = tiy + offseti;
                        auto const y2 = tiy + offsetj;

                        // get element indices
                        auto const i = x1;
                        auto const j = (tix + k) % nb_max + offsetj;
                        if(i < n && j < n)
                        {
                            auto const c = sh_cosines[tix];
                            auto const s1 = sh_sines[tix];
                            auto const s2 = conj(s1);
                            // apply J' from the left
                            {
                                auto const temp1 = A(i, y1);
                                auto const temp2 = A(j, y1);
                                A(i, y1) = c * temp1 + s1 * temp2;
                                A(j, y1) = -s2 * temp1 + c * temp2;
                            }

                            if(y2 < n)
                            {
                                auto const temp1 = A(i, y2);
                                auto const temp2 = A(j, y2);
                                A(i, y2) = c * temp1 + s1 * temp2;
                                A(j, y2) = -s2 * temp1 + c * temp2;
                            }
                        }
                    }
                }

                __syncthreads();

                for(auto tiy = tiy_start; tiy < nb_max; tiy += tiy_inc)
                {
                    for(auto tix = tix_start; tix < nb_max; tix += tix_inc)
                    {
                        auto const x1 = tix + offseti;
                        auto const x2 = tix + offsetj;
                        auto const y1 = tiy + offseti;
                        auto const y2 = tiy + offsetj;

                        // get element indices
                        auto const i = x1;
                        auto const j = (tix + k) % nb_max + offsetj;
                        if((tiy == 0) && (j < n))
                        {
                            // round aij and aji to zero
                            A(i, j) = 0;
                            A(j, i) = 0;
                        }
                    }
                }
            } // end for k

            // -----------------------------------
            // write out Jsh to J in device memory
            // -----------------------------------
            if((J_ != nullptr) && use_Jsh)
            {
                __syncthreads();

                auto const i_start = tix_start;
                auto const i_inc = tix_inc;

                auto const j_start = tiy_start;
                auto const j_inc = tiy_inc;

                for(auto j = j_start; j < ncolsJ; j += j_inc)
                {
                    for(auto i = i_start; i < nrowsJ; i += i_inc)
                    {
                        J(i, j) = Jmat(i, j);
                    }
                }
                __syncthreads();
            }
        } // end for ipair
    } // end for bid
}

/** SYEVJ_OFFD_ROTATE rotates off-diagonal blocks using the rotations calculated by SYEVJ_OFFD_KERNEL.

    Call this kernel with batch_count groups in z, 2*BS2 threads in x and BS2/2 threads in y.
    For a matrix consisting of b * b blocks, use b / 2 groups in x and 2(b - 2) groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_rotate_org(const bool skip_block,
                                            const rocblas_int blocks,
                                            const rocblas_int n,
                                            U AA,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            T* JA,
                                            rocblas_int* top,
                                            rocblas_int* bottom,
                                            rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[bix];
    rocblas_int j = bottom[bix];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);
    if(skip_block && (biy / 2 == i || biy / 2 == j))
        return;

    rocblas_int nb_max = hipBlockDim_x / 2;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int offsetx = (tix < nb_max ? offseti : offsetj - nb_max);
    rocblas_int offsety = biy * hipBlockDim_y;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = std::min(n - offsetj, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * 4 * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += J[tix + k * ldj] * A[y + (k + offseti) * lda];
        for(k = 0; k < nb; k++)
            temp += J[tix + (k + nb_max) * ldj] * A[y + (k + offsetj) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += conj(J[tix + k * ldj]) * A[(k + offseti) + y * lda];
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + (k + nb_max) * ldj]) * A[(k + offsetj) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}

template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_rotate(const bool skip_block,
                                        const rocblas_int nb_max,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* JA,
                                        rocblas_int* top,
                                        rocblas_int* bottom,
                                        rocblas_int* completed,
                                        rocblas_int const batch_count)
{
    bool constexpr APPLY_RIGHT = (!APPLY_LEFT);

    auto const blocks = ceil(n, nb_max);
    auto const even_blocks = blocks + (blocks % 2);
    auto const half_blocks = even_blocks / 2;

    auto bsize = [=](auto iblock) {
        auto const nb_last = n - (blocks - 1) * nb_max;
        bool const is_last_block = (iblock == (blocks - 1));
        return ((is_last_block) ? nb_last : nb_max);
    };

    auto const nbx = hipGridDim_x;
    auto const nby = hipGridDim_y;
    auto const nbz = hipGridDim_z;
    auto const nx = hipBlockDim_x;
    auto const ny = hipBlockDim_y;

    auto const i_start = hipThreadIdx_x;
    auto const j_start = hipThreadIdx_y;

    auto const ib_start = hipBlockIdx_x;
    auto const jb_start = hipBlockIdx_y;
    auto const bid_start = hipBlockIdx_z;

    auto const i_inc = nx;
    auto const j_inc = ny;
    auto const ib_inc = nbx;
    auto const jb_inc = nby;
    auto const bid_inc = nbz;

    auto const npairs = half_blocks;
    auto const ipair_start = ib_start;
    auto const ipair_inc = ib_inc;

    auto const kb_start = jb_start;
    auto const kb_inc = jb_inc;

    auto constexpr max_lds = 64 * 1024;
    auto constexpr len_shmem = max_lds / sizeof(T);

    auto const ldj = 2 * nb_max;
    auto const len_Jsh = ldj * (2 * nb_max);
    auto const len_Ash = (2 * nb_max) * nb_max;

    extern __shared__ double lmem[];
    T* pfree = reinterpret_cast<T*>(&(lmem[0]));

    T* const Ash_ = pfree;
    pfree += len_Ash;
    T* const Jsh_ = pfree;
    pfree += len_Jsh;

    // ---------------------------------
    // store J into shared memory only if
    // there is sufficient space
    // ---------------------------------
    bool const use_Jsh = ((len_Jsh + len_Ash) <= len_shmem);

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        if(completed[bid + 1])
        {
            continue;
        };

        T* const __restrict__ A_ = load_ptr_batch<T>(AA, bid, shiftA, strideA);

        for(auto ipair = ipair_start; ipair < npairs; ipair += ipair_inc)
        {
            // ------------------------------------
            // consider blocks in upper triangular
            // want   (iblock < jblock)
            // ------------------------------------
            auto const iblock = std::min(top[ipair], bottom[ipair]);
            auto const jblock = std::max(top[ipair], bottom[ipair]);

            if((iblock >= blocks) || (jblock >= blocks))
            {
                continue;
            }

            auto const offseti = iblock * nb_max;
            auto const offsetj = jblock * nb_max;

            auto const ni = bsize(iblock);
            auto const nj = bsize(jblock);
            auto const nrowsJ = (ni + nj);
            auto const ncolsJ = nrowsJ;

            auto const jid = ipair + bid * half_blocks;
            T const* const __restrict__ J_ = JA + (jid * 4 * nb_max * nb_max);

            // ---------------------------------
            // store J into shared memory only if
            // there is sufficient space
            // ---------------------------------
            T const* const Jmat_ = (use_Jsh) ? Jsh_ : J_;
            auto Jmat = [=](auto i, auto j) -> const T { return (Jmat_[i + j * ldj]); };

            if(use_Jsh)
            {
                // -------------------------
                // load J into shared memory
                // -------------------------

                auto const ij_start = i_start + j_start * nx;
                auto const ij_inc = nx * ny;
                auto const len_Jsh = 4 * nb_max * nb_max;

                __syncthreads();

                for(auto ij = ij_start; ij < len_Jsh; ij += ij_inc)
                {
                    Jsh_[ij] = J_[ij];
                }
                __syncthreads();
            }

            for(auto kb = kb_start; kb < blocks; kb += kb_inc)
            {
                auto const kblock = kb;

                // -----------------------------------------------------------
                // Note iblock and jblock in matrix A has already been updated
                // so skip those blocks when updating matrix A
                // However, those all blocks should be updated for matrix V
                // of eigenvectors
                // -----------------------------------------------------------
                if(skip_block && ((kblock == iblock) || (kblock == jblock)))
                {
                    continue;
                };

                auto const offsetk = kblock * nb_max;

                auto A
                    = [=](auto i, auto j) -> T& { return (A_[i + j * static_cast<int64_t>(lda)]); };

                auto const nk = bsize(kblock);

                auto const nrows_Ash = (APPLY_LEFT) ? nrowsJ : nk;
                auto const ncols_Ash = (APPLY_LEFT) ? nk : nrowsJ;

                auto const ldAsh = nrows_Ash;
                auto Ash = [=](auto i, auto j) -> T& { return (Ash_[i + j * ldAsh]); };

                // -------------------------------
                // expression for global row index
                // -------------------------------
                auto get_ia = [=](auto i) {
                    auto const ia = (APPLY_LEFT) ? ((i < ni) ? (offseti + i) : (offsetj + (i - ni)))
                                                 : (offsetk + i);

                    return (ia);
                };

                // ----------------------------------
                // expression for global column index
                // ----------------------------------
                auto get_ja = [=](auto j) {
                    auto const ja = (APPLY_LEFT) ? (offsetk + j)
                                                 : ((j < ni) ? (offseti + j) : (offsetj + (j - ni)));
                    return (ja);
                };

                {
                    // ------------------------
                    // load A into shared memory
                    // ------------------------
                    __syncthreads();

                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        auto const ja = get_ja(j);

                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            //      -----------------------------------------
                            // Note if (APPLY_LEFT)  [kblock] is the block column
                            //      if (APPLY_RIGHT) [kblock] is the block row
                            //      -----------------------------------------
                            auto const ia = get_ia(i);

                            Ash(i, j) = A(ia, ja);
                        }
                    }
                    __syncthreads();
                }

                // apply J to the current block
                //
                if(APPLY_RIGHT)
                {
                    // ------------
                    // NOTE: J is stored in row-major order in device memory
                    // thus Jsh is also stored in row-major order in shared memory
                    //
                    // A <- Ash * transpose(Jsh)
                    // ------------

                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        auto const ja = get_ja(j);

                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            T aij = 0;
                            for(auto k = 0; k < nrowsJ; k++)
                            {
                                T const aik = Ash(i, k);
                                T const J_kj = Jmat(j, k);
                                aij += aik * J_kj;
                            }

                            auto const ia = get_ia(i);
                            A(ia, ja) = aij;
                        }
                    }

                    __syncthreads();
                }
                else
                {
                    // APLLY_LEFT

                    // --------------------------------
                    // NOTE: J is stored in row-major order in device memory
                    // thus Jsh is also stored in row-major order in shared memory
                    //
                    // A <-  conj( (Jsh) ) * Ash
                    // --------------------------------

                    for(auto j = j_start; j < ncols_Ash; j += j_inc)
                    {
                        auto const ja = get_ja(j);

                        for(auto i = i_start; i < nrows_Ash; i += i_inc)
                        {
                            T aij = 0;
                            for(auto k = 0; k < nrowsJ; k++)
                            {
                                T const Jc_ik = conj(Jmat(i, k));
                                T const akj = Ash(k, j);

                                aij += Jc_ik * akj;
                            }

                            auto const ia = get_ia(i);
                            A(ia, ja) = aij;
                        }
                    }
                    __syncthreads();
                }

            } // end for kb
        } // end for ipair
    } // end for bid
}

/** SYEVJ_CYCLE_PAIRS cycles the block-level top/bottom pairs to progress the sweep.

    Call this kernel with any number of threads in x. (Top/bottom pairs are shared across batch instances,
    so only one thread group is needed.) **/
template <typename T>
ROCSOLVER_KERNEL void
    syevj_cycle_pairs(const rocblas_int half_blocks, rocblas_int* top, rocblas_int* bottom)
{
    rocblas_int n = half_blocks - 1;

    auto cycle = [n = n](auto i) -> auto
    {
        using I = decltype(i);
        i = (i - 1) % (2 * n + 1) + 1;
        I j{};

        if(i % 2 == 0)
        {
            j = i + 2;
            if(j > 2 * n)
            {
                j = 2 * n + 1;
            }
        }
        else
        {
            j = i - 2;
            if(j < 1)
            {
                j = 2;
            }
        }

        return j;
    };

    rocblas_int tidx = hipThreadIdx_x;
    rocblas_int dimx = hipBlockDim_x;

    if(tidx == 0)
    {
        bottom[0] = cycle(bottom[0]);
    }
    for(rocblas_int l = tidx + 1; l < half_blocks; l += dimx)
    {
        top[l] = cycle(top[l]);
        bottom[l] = cycle(bottom[l]);
    }
}

/** SYEVJ_CALC_NORM calculates the residual norm of the matrix.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void syevj_calc_norm(const rocblas_int n,
                                      const rocblas_int sweeps,
                                      S* residual,
                                      T* AcpyA,
                                      S* norms,
                                      rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int dimx = hipBlockDim_x;

    if(completed[bid + 1])
        return;

    // local variables
    rocblas_int i, j;

    // array pointers
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);

    S local_res = 0;
    for(i = tid; i < n; i += dimx)
    {
        for(j = 0; j < i; j++)
            local_res += 2 * std::norm(Acpy[i + j * n]);
    }
    sh_res[tid] = local_res;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < std::min(n, dimx); i++)
            local_res += sh_res[i];

        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = sweeps + 1;
            atomicAdd(completed, 1);
        }
    }
}

/** SYEVJ_FINALIZE sets the output values for SYEVJ, and sorts the eigenvalues and
    eigenvectors by selection sort if applicable.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_finalize(const rocblas_esort esort,
                                     const rocblas_evect evect,
                                     const rocblas_int n,
                                     U AA,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     S* residual,
                                     const rocblas_int max_sweeps,
                                     rocblas_int* n_sweeps,
                                     S* WW,
                                     const rocblas_stride strideW,
                                     rocblas_int* info,
                                     T* AcpyA,
                                     rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, m;
    rocblas_int sweeps = 0;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* W = WW + bid * strideW;
    T* Acpy = AcpyA + bid * n * n;

    // finalize outputs
    if(tid == 0)
    {
        rocblas_int sweeps = completed[bid + 1] - 1;
        residual[bid] = sqrt(residual[bid]);
        if(sweeps >= 0)
        {
            n_sweeps[bid] = sweeps;
            info[bid] = 0;
        }
        else
        {
            n_sweeps[bid] = max_sweeps;
            info[bid] = 1;
        }
    }

    // put eigenvalues into output array
    for(i = tid; i < n; i += hipBlockDim_x)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if((evect == rocblas_evect_none && tid > 0) || esort == rocblas_esort_none)
        return;

    // sort eigenvalues & vectors
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += hipBlockDim_x)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
            }
        }
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevj_heevj_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_Acpy,
                                         size_t* size_J,
                                         size_t* size_norms,
                                         size_t* size_top,
                                         size_t* size_bottom,
                                         size_t* size_completed)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_Acpy = 0;
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
        return;
    }

    // size of temporary workspace for copying A
    *size_Acpy = sizeof(T) * n * n * batch_count;

    if(n <= SYEVJ_BLOCKED_SWITCH)
    {
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
        return;
    }

    rocblas_int const nb_max = BS2;
    auto const half_n = ceil(n, 2);
    auto const blocks = ceil(n, nb_max);
    auto const half_blocks = ceil(blocks, 2);

    // size of temporary workspace to store the block rotation matrices
    if(half_blocks == 1 && evect == rocblas_evect_none)
    {
        *size_J = sizeof(T) * blocks * nb_max * nb_max * batch_count;
    }
    else
    {
        *size_J = sizeof(T) * half_blocks * 4 * nb_max * nb_max * batch_count;
    }

    // size of temporary workspace to store the full matrix norm
    *size_norms = sizeof(S) * batch_count;

    // size of arrays for temporary top/bottom pairs
    *size_top = sizeof(rocblas_int) * half_blocks * batch_count;
    *size_bottom = sizeof(rocblas_int) * half_blocks * batch_count;

    // size of temporary workspace to indicate problem completion
    *size_completed = sizeof(rocblas_int) * (batch_count + 1);
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevj_heevj_argCheck(rocblas_handle handle,
                                              const rocblas_esort esort,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(esort != rocblas_esort_none && esort != rocblas_esort_ascending)
        return rocblas_status_invalid_value;
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || max_sweeps <= 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_template(rocblas_handle handle,
                                              const rocblas_esort esort,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              const S abstol,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              const rocblas_stride strideW,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* Acpy,
                                              T* J,
                                              S* norms,
                                              rocblas_int* top,
                                              rocblas_int* bottom,
                                              rocblas_int* completed)
{
    ROCSOLVER_ENTER("syevj_heevj", "esort:", esort, "evect:", evect, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n <= 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        // scalar case
        if(n == 1)
            ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream, evect, A,
                                    strideA, W, strideW, batch_count);

        return rocblas_status_success;
    }

    // absolute tolerance for evaluating when the algorithm has converged
    S eps = get_epsilon<S>();
    S atol = (abstol <= 0 ? eps : abstol);

    // local variables
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    if(n <= SYEVJ_BLOCKED_SWITCH)
    {
        // *** USE SINGLE SMALL-SIZE KERNEL ***
        // (TODO: SYEVJ_BLOCKED_SWITCH may need re-tuning as it could be larger than 64 now).

        rocblas_int ddx, ddy;
        syevj_get_dims(n, SYEVJ_BDIM, &ddx, &ddy);
        dim3 grid(1, 1, batch_count);
        dim3 threads(ddx * ddy, 1, 1);
        size_t lmemsize = (sizeof(S) + sizeof(T)) * ddx + 2 * sizeof(rocblas_int) * half_n;

        ROCSOLVER_LAUNCH_KERNEL(syevj_small_kernel<T>, grid, threads, lmemsize, stream, esort,
                                evect, uplo, n, A, shiftA, lda, strideA, atol, eps, residual,
                                max_sweeps, n_sweeps, W, strideW, info, Acpy);
    }
    else
    {
        // use original algorithm for small problems
        auto const n_threshold = 1024;

        bool const use_offd_kernel_org = (n <= n_threshold);
        bool const use_diag_rotate_org = (n <= n_threshold);
        bool const use_offd_rotate_org = (n <= n_threshold);

        // *** USE BLOCKED KERNELS ***
        rocblas_int const nb_max = BS2;

        // kernel dimensions
        rocblas_int const blocksReset = batch_count / BS1 + 1;
        auto const blocks = ceil(n, nb_max);
        auto const even_blocks = blocks + (blocks % 2);
        auto const half_blocks = even_blocks / 2;

        dim3 gridReset(blocksReset, 1, 1);
        dim3 grid(1, batch_count, 1);
        dim3 gridDK(blocks, 1, batch_count);
        dim3 gridDR(blocks, blocks, batch_count);
        dim3 gridOK(half_blocks, 1, batch_count);

        dim3 gridPairs(1, 1, 1);
        dim3 threadsReset(BS1, 1, 1);
        dim3 threads(BS1, 1, 1);
        dim3 threadsDK(BS2 / 2, BS2 / 2, 1);
        dim3 threadsDR(BS2, BS2, 1);
        dim3 threadsOK(BS2, BS2, 1);

        dim3 gridOR_org(half_blocks, 2 * blocks, batch_count);
        dim3 threadsOR_org(2 * BS2, BS2 / 2, 1);

        dim3 gridOR_new(half_blocks, std::max(1, blocks / 4), batch_count);
        dim3 threadsOR_new(BS2, BS2, 1);

        dim3 gridOR = (use_offd_rotate_org) ? gridOR_org : gridOR_new;
        dim3 threadsOR = (use_offd_rotate_org) ? threadsOR_org : threadsOR_new;

        size_t const size_lds = 64 * 1024;
        // shared memory sizes
        size_t const lmemsizeInit = 2 * sizeof(S) * BS1;
        size_t const lmemsizeDK = (sizeof(S) + sizeof(T) + 2 * sizeof(rocblas_int)) * (BS2 / 2);

        size_t lmemsizeDR = std::min(size_lds, 2 * sizeof(T) * nb_max * nb_max);
        {
            size_t const size_Ash = sizeof(T) * nb_max * nb_max;
            size_t const size_Jsh = sizeof(T) * nb_max * nb_max;
            bool const use_Jsh = ((size_Ash + size_Jsh) <= size_lds);
            lmemsizeDR = (use_Jsh) ? (size_Ash + size_Jsh) : size_Ash;

            assert(size_Ash <= size_lds);
        }

        size_t lmemsizeOK = (sizeof(S) + sizeof(T)) * nb_max;
        {
            // ---------------------------------------------------------
            // try to hold J in shared memory if there is sufficient LDS
            // ---------------------------------------------------------
            size_t const size_sh_cosines = sizeof(S) * nb_max;
            size_t const size_sh_sines = sizeof(T) * nb_max;
            size_t const size_Jsh = sizeof(T) * (2 * nb_max) * (2 * nb_max);
            bool const use_Jsh = (size_sh_cosines + size_sh_sines + size_Jsh) <= size_lds;
            lmemsizeOK = (use_Jsh) ? (size_sh_cosines + size_sh_sines + size_Jsh)
                                   : (size_sh_cosines + size_sh_sines);
        }

        size_t const lmemsizePairs
            = ((half_blocks > BS1) ? 2 * sizeof(rocblas_int) * half_blocks : 0);

        // ------------------------------------------
        // store 2x2 block J and 2x1 block of A in LDS if
        // there is sufficient space,
        // otherwise store only 2x1 block of A in LDS
        // ------------------------------------------

        size_t lmemsizeOR = size_lds;
        {
            size_t const size_Ash = sizeof(T) * 2 * nb_max * nb_max;
            size_t const size_Jsh = sizeof(T) * 4 * nb_max * nb_max;
            lmemsizeOR = ((size_Ash + size_Jsh) <= size_lds) ? (size_Ash + size_Jsh) : size_Ash;

            assert(size_Ash <= size_lds);
        }

        bool const ev = (evect != rocblas_evect_none);
        rocblas_int h_sweeps = 0;
        rocblas_int h_completed = 0;

        // set completed = 0
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, completed,
                                batch_count + 1, 0);

        // copy A to Acpy, set A to identity (if applicable), compute initial residual, and
        // initialize top/bottom pairs (if applicable)
        ROCSOLVER_LAUNCH_KERNEL(syevj_init<T>, grid, threads, lmemsizeInit, stream, evect, uplo,
                                half_blocks, n, A, shiftA, lda, strideA, atol, residual, Acpy,
                                norms, top, bottom, completed);

        while(h_sweeps < max_sweeps)
        {
            // if all instances in the batch have finished, exit the loop
            HIP_CHECK(hipMemcpyAsync(&h_completed, completed, sizeof(rocblas_int),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            if(h_completed == batch_count)
                break;

            // decompose diagonal blocks
            ROCSOLVER_LAUNCH_KERNEL(syevj_diag_kernel<T>, gridDK, threadsDK, lmemsizeDK, stream, n,
                                    Acpy, 0, n, n * n, eps, J, completed);

            // apply rotations calculated by diag_kernel
            if(use_diag_rotate_org)
            {
                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate_org<false, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, true, n, Acpy, 0, n, n * n, J, completed);

                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate_org<true, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, true, n, Acpy, 0, n, n * n, J, completed);
            }
            else
            {
                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, true, nb_max, n, Acpy, 0, n, n * n, J,
                                        completed, batch_count);

                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<true, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, true, nb_max, n, Acpy, 0, n, n * n, J,
                                        completed, batch_count);
            }

            // update eigenvectors
            if(ev)
            {
                if(use_diag_rotate_org)
                {
                    ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate_org<false, T, S>), gridDR, threadsDR,
                                            lmemsizeDR, stream, false, n, A, shiftA, lda, strideA,
                                            J, completed);
                }
                else
                {
                    ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR,
                                            lmemsizeDR, stream, false, nb_max, n, A, shiftA, lda,
                                            strideA, J, completed, batch_count);
                }
            }

            if(half_blocks == 1)
            {
                // decompose off-diagonal block
                if(use_offd_kernel_org)
                {
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel_org<T, S>), gridOK, threadsOK,
                                            lmemsizeOK, stream, blocks, n, Acpy, 0, n, n * n, eps,
                                            (ev ? J : nullptr), top, bottom, completed);
                }
                else
                {
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK,
                                            lmemsizeOK, stream, nb_max, n, Acpy, 0, n, n * n, eps,
                                            (ev ? J : nullptr), top, bottom, completed, batch_count);
                }

                // update eigenvectors
                if(ev)
                {
                    if(use_offd_rotate_org)
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate_org<false, T, S>), gridOR,
                                                threadsOR, lmemsizeOR, stream, false, blocks, n, A,
                                                shiftA, lda, strideA, J, top, bottom, completed);
                    }
                    else
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR,
                                                lmemsizeOR, stream, false, nb_max, n, A, shiftA, lda,
                                                strideA, J, top, bottom, completed, batch_count);
                    }
                }
            }
            else
            {
                for(rocblas_int b = 0; b < even_blocks - 1; b++)
                {
                    // decompose off-diagonal blocks, indexed by top/bottom pairs
                    if(use_offd_kernel_org)
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel_org<T, S>), gridOK, threadsOK,
                                                lmemsizeOK, stream, blocks, n, Acpy, 0, n, n * n,
                                                eps, J, top, bottom, completed);
                    }
                    else
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK,
                                                lmemsizeOK, stream, nb_max, n, Acpy, 0, n, n * n,
                                                eps, J, top, bottom, completed, batch_count);
                    }

                    // apply rotations calculated by offd_kernel
                    if(use_offd_rotate_org)
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate_org<false, T, S>), gridOR,
                                                threadsOR, lmemsizeOR, stream, true, blocks, n,
                                                Acpy, 0, n, n * n, J, top, bottom, completed);
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate_org<true, T, S>), gridOR,
                                                threadsOR, lmemsizeOR, stream, true, blocks, n,
                                                Acpy, 0, n, n * n, J, top, bottom, completed);
                    }
                    else
                    {
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR,
                                                lmemsizeOR, stream, true, nb_max, n, Acpy, 0, n,
                                                n * n, J, top, bottom, completed, batch_count);
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<true, T, S>), gridOR, threadsOR,
                                                lmemsizeOR, stream, true, nb_max, n, Acpy, 0, n,
                                                n * n, J, top, bottom, completed, batch_count);
                    }

                    // update eigenvectors
                    if(ev)
                    {
                        if(use_offd_rotate_org)
                        {
                            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate_org<false, T, S>), gridOR,
                                                    threadsOR, lmemsizeOR, stream, false, blocks, n,
                                                    A, shiftA, lda, strideA, J, top, bottom,
                                                    completed);
                        }
                        else
                        {
                            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR,
                                                    threadsOR, lmemsizeOR, stream, false, nb_max, n,
                                                    A, shiftA, lda, strideA, J, top, bottom,
                                                    completed, batch_count);
                        }
                    }

                    // cycle top/bottom pairs
                    ROCSOLVER_LAUNCH_KERNEL(syevj_cycle_pairs<T>, gridPairs, threads, lmemsizePairs,
                                            stream, half_blocks, top, bottom);
                }
            }

            // compute new residual
            h_sweeps++;
            ROCSOLVER_LAUNCH_KERNEL(syevj_calc_norm<T>, grid, threads, lmemsizeInit, stream, n,
                                    h_sweeps, residual, Acpy, norms, completed);
        }

        // set outputs and sort eigenvalues & vectors
        ROCSOLVER_LAUNCH_KERNEL(syevj_finalize<T>, grid, threads, 0, stream, esort, evect, n, A,
                                shiftA, lda, strideA, residual, max_sweeps, n_sweeps, W, strideW,
                                info, Acpy, completed);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * *************************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/**************************************************************************************/
/***************** Kernels/Device functions *******************************************/
/**************************************************************************************/

/***** Kernel to compute column of Y *****/
/*****************************************/
// (grid = dim3(ceil(mm / DIM_X), 1, batch_count), block = dim3(DIM_X, DIM_Y))
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(DIM_X* DIM_Y)
    lahr2_computeY_kernel(const rocblas_int mm,
                          const rocblas_int k,
                          const rocblas_int c,
                          U __restrict__ AA,
                          const rocblas_stride shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          T* __restrict__ YA,
                          const rocblas_stride shiftY,
                          const rocblas_int ldy,
                          const rocblas_stride strideY,
                          T* __restrict__ FA,
                          const rocblas_stride shiftF,
                          const rocblas_int ldf,
                          const rocblas_stride strideF,
                          T* __restrict__ tauA,
                          const rocblas_stride strideT)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * DIM_X;
    int totalthsc = groupsc * DIM_Y;
    int idc = bidc * DIM_Y + tidc;
    int idr = bidr * DIM_X + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Y = load_ptr_batch<T>(YA, bid, shiftY, strideY);
    T* F = load_ptr_batch<T>(FA, bid, shiftF, strideF);
    T* tau = tauA + bid * strideT;

    /* ------------------------
    formulate gemv problem:

        components:
            y  = Y(k:mm-1, c)
            A1 = A(k:mm-1, c+1:mm-k)
            A2 = Y(k:mm-1, 0:c-1)
            x1 = A(k+c:mm-1, c)
            x2 = F(0:c-1, c)
            t  = tau(c)

        operation:
            y = t * (A1 * x1 - A2 * x2)
    ------------------------ */
    int m = mm - k;
    int n1 = mm - k - c;
    int n2 = c;
    T* y = Y + idx2D(k, c, ldy);
    T* A1 = A + idx2D(k, c + 1, lda);
    int lda1 = lda;
    T* A2 = Y + idx2D(k, 0, ldy);
    int lda2 = ldy;
    T* x1 = A + idx2D(k + c, c, lda);
    T* x2 = F + idx2D(0, c, ldf);
    T* t = tau + c;

    // Registers/LDS:
    // ac, acs -> accumulator
    __shared__ T acs[DIM_X * DIM_Y];
    T ac;

    for(int i = idr; i < m; i += totalthsr)
    {
        ac = 0;

        for(int j = idc; j < n1; j += totalthsc)
        {
            // A1 * x1
            ac += A1[i + j * size_t(lda1)] * x1[j];
        }

        for(int j = idc; j < n2; j += totalthsc)
        {
            // A2 * x2
            ac -= A2[i + j * size_t(lda2)] * x2[j];
        }

        acs[tidr + tidc * DIM_X] = ac;
        __syncthreads();

        // group reduction
        for(int r = DIM_Y / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * DIM_X];
                acs[tidr + tidc * DIM_X] = ac;
            }
            __syncthreads();
        }

        if(tidc == 0 && i < m)
            y[i] = ac * t[0];
    }
}

/***** Column of Y for large matrices, in two passes *****/
/*********************************************************/
// For m >= LAHR2_SPLIT_MIN_ROWS, the product A1 * x1 of lahr2_computeY_kernel is split over
// nsplit chunks of the columns of A1 (grid = dim3(ceil(m / DIM_X), nsplit, batch_count),
// block = dim3(DIM_X, DIM_Y)), so that all the compute units are busy when m is small
// compared with their number; the partial sums are stored in the workspace P (m-by-nsplit
// per matrix), and lahr2_computeY_sum_kernel adds them (in a fixed order, so that the result
// is deterministic), subtracts A2 * x2 and scales by t.
/** LAHR2_LOAD_NT loads an entry with non-temporal loads (the entries of the trailing matrix
    are read once per column product, so they need not stay in the caches; this lets the
    product read memory a few percent faster). **/
template <typename T>
__device__ __forceinline__ T lahr2_load_nt(const T* p)
{
    if constexpr(rocblas_is_complex<T>)
    {
        using S = decltype(std::real(T{}));
        const S* q = reinterpret_cast<const S*>(p);
        return T(__builtin_nontemporal_load(q), __builtin_nontemporal_load(q + 1));
    }
    else
        return __builtin_nontemporal_load(p);
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(DIM_X* DIM_Y)
    lahr2_computeY_part_kernel(const rocblas_int mm,
                               const rocblas_int k,
                               const rocblas_int c,
                               const rocblas_int chunk,
                               U __restrict__ AA,
                               const rocblas_stride shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* __restrict__ P,
                               const rocblas_stride strideP)
{
    const int bid = hipBlockIdx_z;
    const int tidr = hipThreadIdx_x;
    const int tidc = hipThreadIdx_y;
    const int i = hipBlockIdx_x * DIM_X + tidr;
    const int s = hipBlockIdx_y;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);

    // A1 = A(k:mm-1, c+1:mm-k), x1 = A(k+c:mm-1, c); partial sum over the columns
    // j0:j1-1 of A1, stored in column s of P
    const int m = mm - k;
    const int n1 = mm - k - c;
    const T* A1 = A + idx2D(k, c + 1, lda);
    const T* x1 = A + idx2D(k + c, c, lda);
    const int j0 = s * chunk;
    const int j1 = std::min(n1, j0 + chunk);

    __shared__ T acs[DIM_X * DIM_Y];
    T ac = 0;
    if(i < m)
        for(int j = j0 + tidc; j < j1; j += DIM_Y)
            ac += lahr2_load_nt(A1 + i + j * size_t(lda)) * x1[j];
    acs[tidr + tidc * DIM_X] = ac;
    __syncthreads();

    // group reduction
    for(int r = DIM_Y / 2; r > 0; r /= 2)
    {
        if(tidc < r)
            acs[tidr + tidc * DIM_X] += acs[tidr + (tidc + r) * DIM_X];
        __syncthreads();
    }

    if(tidc == 0 && i < m)
        P[bid * strideP + s * size_t(m) + i] = acs[tidr];
}

template <typename T>
ROCSOLVER_KERNEL void lahr2_computeY_sum_kernel(const rocblas_int mm,
                                                const rocblas_int k,
                                                const rocblas_int c,
                                                const rocblas_int nsplit,
                                                const T* __restrict__ P,
                                                const rocblas_stride strideP,
                                                T* __restrict__ YA,
                                                const rocblas_stride shiftY,
                                                const rocblas_int ldy,
                                                const rocblas_stride strideY,
                                                T* __restrict__ FA,
                                                const rocblas_stride shiftF,
                                                const rocblas_int ldf,
                                                const rocblas_stride strideF,
                                                T* __restrict__ tauA,
                                                const rocblas_stride strideT)
{
    const int bid = hipBlockIdx_z;
    const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T* Y = load_ptr_batch<T>(YA, bid, shiftY, strideY);
    T* F = load_ptr_batch<T>(FA, bid, shiftF, strideF);
    T* tau = tauA + bid * strideT;

    // y = Y(k:mm-1, c) = t * (sum of the partials - A2 * x2), with A2 = Y(k:mm-1, 0:c-1),
    // x2 = F(0:c-1, c) and t = tau(c)
    const int m = mm - k;
    if(i < m)
    {
        const T* Pb = P + bid * strideP;
        T ac = 0;
        for(int s = 0; s < nsplit; s++)
            ac += Pb[i + s * size_t(m)];
        const T* A2 = Y + idx2D(k, 0, ldy);
        const T* x2 = F + idx2D(0, c, ldf);
        for(int j = 0; j < c; j++)
            ac -= A2[i + j * size_t(ldy)] * x2[j];
        Y[idx2D(k + i, c, ldy)] = ac * tau[c];
    }
}

/***** Scale current column and set diag of triangular factor kernel *****/
/*************************************************************************/
template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lahr2_scale_set_tau(const I j,
                                                                      U __restrict__ FA,
                                                                      const rocblas_stride shiftF,
                                                                      const rocblas_stride strideF,
                                                                      T* __restrict__ tauA,
                                                                      const rocblas_stride strideT)
{
    const auto bid = blockIdx.z;
    const auto tid = threadIdx.x;

    // select batch instance
    T* F = load_ptr_batch<T>(FA, bid, shiftF, strideF);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideT);

    /* ------------------------
    formulate problem:

        F[ 0:j-1 ] *= -tau;
        F[ j ] = tau;
    ------------------------ */

    const T t = *tau;

    for(I i = tid; i < j; i += MAX_THDS)
    {
        F[i] *= -t;
    }

    if(tid == 0)
    {
        F[j] = t;
    }
}

template <bool BATCHED, typename T>
void rocsolver_lahr2_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int nb,
                                   const rocblas_int batch_count,
                                   size_t* size_work_workArr,
                                   size_t* size_norms,
                                   size_t* size_work_vec,
                                   size_t* size_beta)
{
    // if quick return no workspace needed
    if(n <= 1 || nb == 0 || batch_count == 0)
    {
        *size_work_workArr = 0;
        *size_norms = 0;
        *size_work_vec = 0;
        *size_beta = 0;
        return;
    }

    size_t s1, s2;

    // size of array of pointers (batched cases)
    if(BATCHED)
        s1 = 2 * sizeof(T*) * batch_count;
    else
        s1 = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n - k, batch_count, &s2, size_norms);

    // partial sums of the products with the trailing matrix, for large matrices (see
    // lahr2_computeY_part_kernel); norms is free after larfg
    if(n - k >= LAHR2_SPLIT_MIN_ROWS)
        *size_norms = std::max(*size_norms, sizeof(T) * n * LAHR2_MAX_SPLIT * batch_count);

    // work_workArr also used as trmv scratch (length nb per batch)
    *size_work_workArr = std::max({s1, s2, sizeof(T) * nb * batch_count});

    // separate w vector buffer (length nb per batch) for the update step trmv operations,
    // kept separate from F to avoid rocblas aliasing checks
    *size_work_vec = sizeof(T) * nb * batch_count;

    // beta: one scalar per batch instance
    *size_beta = sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lahr2_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        const rocblas_int ldf,
                                        const rocblas_int ldy,
                                        U A,
                                        T* tau,
                                        T* F,
                                        T* Y,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    // n=0 or n=1: quick return, not an error
    if(n < 0 || k < 1 || nb < 0 || (n > 1 && (nb < 1 || k >= n || nb > n - k))
       || lda < std::max(1, n) || ldf < nb || ldy < std::max(1, n) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (!tau) || (!F) || (!Y))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_lahr2_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int nb,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F, /* Alias for the triangular factor T */
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        T* Y,
                                        const rocblas_stride shiftY,
                                        const rocblas_int ldy,
                                        const rocblas_stride strideY,
                                        const rocblas_int batch_count,
                                        void* work_workArr,
                                        T* norms,
                                        T* work_vec,
                                        T* beta)
{
    ROCSOLVER_ENTER("lahr2", "n:", n, "k:", k, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "ldf:", ldf, "shiftY:", shiftY, "ldy:", ldy, "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return
    if(n <= 1 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // number of compute units (for the split of the products with A, see below)
    int device, ncu;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, device));

    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    // host scalars
    const T one = 1;
    const T zero = 0;
    const T neg_one = -1;

    // work_vec: dedicated length-nb w vector per batch for the update step.
    // stride_work: per-batch stride for both work_vec and the trmv scratch (work_workArr).
    rocblas_stride stride_work = rocblas_stride(nb);

    // beta = A(k+j, j) saved before larfg sets it to 1, restored at
    // the start of the next iteration (or after the loop for j=nb-1).
    // stride_beta: per-batch stride for beta.
    rocblas_stride stride_beta = rocblas_stride(1);

    // ------------------------------------------------------------------------
    //                                Main loop
    // ------------------------------------------------------------------------
    for(rocblas_int j = 0; j < nb; ++j)
    {
        if(j > 0)
        {
            // ----------------------------------------------------------------
            // Update A(k:n-1, j):
            //
            // (a) A(k:n-1, j) -= Y(k:n-1, 0:j-1) * A(k+j-1, 0:j-1)^H
            // ----------------------------------------------------------------
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(k + j - 1, 0, lda), lda,
                                            strideA, batch_count);

            rocblasCall_gemv<T>(
                handle, rocblas_operation_none, n - k, j, &neg_one, 0, Y, shiftY + idx2D(k, 0, ldy),
                ldy, strideY, A, shiftA + idx2D(k + j - 1, 0, lda), lda, strideA, &one, 0, A,
                shiftA + idx2D(k, j, lda), 1, strideA, batch_count, (T**)work_workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(k + j - 1, 0, lda), lda,
                                            strideA, batch_count);

            // ----------------------------------------------------------------
            // (b) Apply (I - V*T^H*V^H) to A(k:n-1, j) from the left.
            //     Uses work_vec as workspace w (length j).
            //
            //     V = A(k:n-1, 0:j-1), lower trapezoidal matrix where
            //       V1 = A(k:k+j-1, 0:j-1)   -- unit lower triangular
            //       V2 = A(k+j:n-1, 0:j-1)
            //
            //     w  = V^H * b (TZMV lower ^H unit)
            //     w  = T^H * w (TRMV upper ^H non-unit)
            //     b -= V * w   (TZMV lower unit)
            // ----------------------------------------------------------------

            // w = V^H * b
            constexpr rocblas_int TZMVT_DIM_X = BS1;
            ROCSOLVER_LAUNCH_KERNEL((rocsolver_tzmvt_kernel<TZMVT_DIM_X, true, true, true, T>),
                                    dim3(j, 1, batch_count), dim3(TZMVT_DIM_X), 0, stream, n - k, j,
                                    one, A, shiftA + idx2D(k, 0, lda), lda, strideA, A,
                                    shiftA + idx2D(k, j, lda), 1, strideA, zero, work_vec, 0, 1,
                                    stride_work);

            // w = T^H * w  (TRMV upper ^H non-unit)
            rocblasCall_trmv<T>(handle, rocblas_fill_upper, rocblas_operation_conjugate_transpose,
                                rocblas_diagonal_non_unit, j, F, 0, ldf, strideF, work_vec, 0, 1,
                                stride_work, (T*)work_workArr, stride_work, batch_count);

            // b -= V * w
            constexpr rocblas_int TZMVN_DIM_X = 64;
            constexpr rocblas_int TZMVN_DIM_Y = ROCSOLVER_ASAN_VALUE(4, 16);
            ROCSOLVER_LAUNCH_KERNEL((rocsolver_tzmvn_kernel<TZMVN_DIM_X, TZMVN_DIM_Y, true, true, T>),
                                    dim3((n - k - 1) / TZMVN_DIM_X + 1, 1, batch_count),
                                    dim3(TZMVN_DIM_X, TZMVN_DIM_Y), 0, stream, n - k, j, neg_one, A,
                                    shiftA + idx2D(k, 0, lda), lda, strideA, work_vec, 0, 1,
                                    stride_work, one, A, shiftA + idx2D(k, j, lda), 1, strideA);

            // Restore A(k+j-1, j-1) = beta saved from the previous iteration
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                    stream, (S*)beta, 0, stride_beta, A,
                                    shiftA + idx2D(k + j - 1, j - 1, lda), lda, strideA,
                                    (rocblas_int)1);
        }

        // --------------------------------------------------------------------
        // Generate Householder reflector H(j+1) to annihilate A(k+j+1:n-1, j)
        //  - Set beta = A(k+j, j), then set A(k+j, j) = 1
        // --------------------------------------------------------------------
        rocsolver_larfg_template(handle, n - k - j, A, shiftA + idx2D(k + j, j, lda), (S*)beta, 0,
                                 stride_beta, A, shiftA + idx2D(std::min(k + j + 1, n - 1), j, lda),
                                 1, strideA, tau + j, strideT, batch_count, (T*)work_workArr, norms);

        // --------------------------------------------------------------------
        // Compute Y(k:n-1, j)
        // --------------------------------------------------------------------
        if(j > 0)
        {
            // T(0:j-1, j) = A(k+j:n-1, 0:j-1)^H * A(k+j:n-1, j)
            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - k - j, j, &one,
                                0, A, shiftA + idx2D(k + j, 0, lda), lda, strideA, A,
                                shiftA + idx2D(k + j, j, lda), 1, strideA, &zero, 0, F,
                                idx2D(0, j, ldf), 1, strideF, batch_count, (T**)work_workArr);
        }

        // Y(k:n-1, j) = t * (A(k:n-1, j+1:n-k) * A(k+j:n-1, j) - Y(k:n-1, 0:j-1) * T(0:j-1, j))
        // (for large matrices, the product with A is split over the columns, with the partial
        // sums in the workspace norms, which is free after larfg, to use all the compute units)
        const rocblas_int m = n - k;
        if(m >= LAHR2_SPLIT_MIN_ROWS)
        {
            constexpr rocblas_int PART_DIM_X = 64;
            constexpr rocblas_int PART_DIM_Y = 4;
            const rocblas_int n1 = m - j;
            const rocblas_int rowblocks = (m - 1) / PART_DIM_X + 1;
            rocblas_int nsplit = (LAHR2_SPLIT_BLOCKS_PER_CU * ncu + rowblocks - 1) / rowblocks;
            nsplit = std::max(rocblas_int(1), std::min({nsplit, n1, rocblas_int(LAHR2_MAX_SPLIT)}));
            const rocblas_int chunk = (n1 - 1) / nsplit + 1;
            nsplit = (n1 - 1) / chunk + 1;
            const rocblas_stride strideP = rocblas_stride(n) * LAHR2_MAX_SPLIT;
            ROCSOLVER_LAUNCH_KERNEL((lahr2_computeY_part_kernel<PART_DIM_X, PART_DIM_Y, T>),
                                    dim3(rowblocks, nsplit, batch_count),
                                    dim3(PART_DIM_X, PART_DIM_Y), 0, stream, n, k, j, chunk, A,
                                    shiftA, lda, strideA, norms, strideP);
            ROCSOLVER_LAUNCH_KERNEL((lahr2_computeY_sum_kernel<T>),
                                    dim3((m - 1) / BS1 + 1, 1, batch_count), dim3(BS1), 0, stream,
                                    n, k, j, nsplit, (const T*)norms, strideP, Y, shiftY, ldy,
                                    strideY, F, 0, ldf, strideF, tau, strideT);
        }
        else
        {
            constexpr rocblas_int COMPY_DIM_X = BS2;
            constexpr rocblas_int COMPY_DIM_Y = BS2;
            ROCSOLVER_LAUNCH_KERNEL((lahr2_computeY_kernel<COMPY_DIM_X, COMPY_DIM_Y, T>),
                                    dim3((n - 1) / COMPY_DIM_X + 1, 1, batch_count),
                                    dim3(COMPY_DIM_X, COMPY_DIM_Y), 0, stream, n, k, j, A, shiftA,
                                    lda, strideA, Y, shiftY, ldy, strideY, F, 0, ldf, strideF, tau,
                                    strideT);
        }

        // --------------------------------------------------------------------
        // Compute T(0:j, j)
        // --------------------------------------------------------------------

        // T(0:j-1, j) *= -tau(j) and T(j, j) = tau(j)
        ROCSOLVER_LAUNCH_KERNEL((lahr2_scale_set_tau<BS1, T>), dim3(1, 1, batch_count),
                                dim3(BS1, 1, 1), 0, stream, j, F, idx2D(0, j, ldf), strideF,
                                tau + j, strideT);
        if(j > 0)
        {
            // T(0:j-1, j) = T(0:j-1, 0:j-1) * T(0:j-1, j)  (upper non-unit)
            rocblasCall_trmv<T>(handle, rocblas_fill_upper, rocblas_operation_none,
                                rocblas_diagonal_non_unit, j, F, 0, ldf, strideF, F, idx2D(0, j, ldf),
                                1, strideF, (T*)work_workArr, stride_work, batch_count);
        }
    }

    // Restore A(k+nb-1, nb-1) = beta
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            (S*)beta, 0, stride_beta, A, shiftA + idx2D(k + nb - 1, nb - 1, lda),
                            lda, strideA, (rocblas_int)1);

    // ------------------------------------------------------------------------
    // Compute Y(0:k-1, 0:nb-1)  (top k rows of Y)
    // ------------------------------------------------------------------------
    if(k > 0)
    {
        // rocblas_internal_trmm_template may recursively call rocblas_internal_gemm_64 with
        // host-side beta constants (&beta_1<T>). If the handle is in device pointer mode those
        // host addresses are misinterpreted as device pointers, corrupting results.

        // Y(0:k-1,:) = A(0:k-1, 1:nb) * A(k:k+nb-1, 0:nb-1)  (right trmm, lower unit)
        rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                            rocblas_diagonal_unit, k, nb, &one, 0, A, shiftA + idx2D(k, 0, lda),
                            lda, strideA, A, shiftA + idx2D(0, 1, lda), lda, strideA, Y, shiftY,
                            ldy, strideY, batch_count, (T**)work_workArr);

        if(n > k + nb)
        {
            // Y(0:k-1,:) += A(0:k-1, nb+1:n-1) * A(k+nb:n-1, 0:nb-1)
            rocsolver_gemm<T>(handle, rocblas_operation_none, rocblas_operation_none, k, nb,
                              n - k - nb, &one, A, shiftA + idx2D(0, nb + 1, lda), lda, strideA, A,
                              shiftA + idx2D(k + nb, 0, lda), lda, strideA, &one, Y, shiftY, ldy,
                              strideY, batch_count, (T**)work_workArr);
        }

        // Y(0:k-1,:) *= T  (right trmm: Y = Y * T, upper non-unit)
        rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                            rocblas_diagonal_non_unit, k, nb, &one, 0, F, 0, ldf, strideF, Y,
                            shiftY, ldy, strideY, batch_count, (T**)work_workArr);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

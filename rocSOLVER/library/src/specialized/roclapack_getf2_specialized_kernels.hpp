/************************************************************************
 * Small kernel algorithm based on:
 * Abdelfattah, A., Haidar, A., Tomov, S., & Dongarra, J. (2017).
 * Factorization and inversion of a million matrices using GPUs: Challenges
 * and countermeasures. Procedia Computer Science, 108, 606-615.
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

/** getf2_small_kernel takes care of of matrices with m < n
    m <= GETF2_MAX_THDS and n <= GETF2_MAX_COLS **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_small_kernel(const I m,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       I* ipivA,
                       const rocblas_stride shiftP,
                       const rocblas_stride strideP,
                       INFO* infoA,
                       const I batch_count,
                       const I offset,
                       I* permut_idx,
                       const rocblas_stride stridePI)
{
    using S = decltype(std::real(T{}));

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    I* ipiv = load_ptr_batch<I>(ipivA, id, shiftP, strideP);
    I* permut = (permut_idx != nullptr ? permut_idx + id * stridePI : nullptr);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = reinterpret_cast<T*>(lmem);
    common += ty * std::max(m, DIM);

    // local variables
    T pivot_value;
    T test_value;
    I pivot_index;
    I mypiv = myrow + 1; // to build ipiv
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j * lda];

        // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        // share current column
        common[myrow] = rA[k];
        __syncthreads();

        // search pivot index
        pivot_index = k;
        pivot_value = common[k];
        for(I i = k + 1; i < m; ++i)
        {
            test_value = common[i];
            if(aabs<S>(pivot_value) < aabs<S>(test_value))
            {
                pivot_value = test_value;
                pivot_index = i;
            }
        }

        // check singularity and scale value for current column
        if(pivot_value != T(0))
            pivot_value = S(1) / pivot_value;
        else if(myinfo == 0)
            myinfo = k + 1;

        // swap rows (lazy swaping)
        if(myrow == pivot_index)
        {
            myrow = k;
            // share pivot row
            for(I j = k + 1; j < DIM; ++j)
                common[j] = rA[j];
        }
        else if(myrow == k)
        {
            myrow = pivot_index;
            mypiv = pivot_index + 1;
            if(permut_idx && pivot_index != k)
                swap(permut[k], permut[pivot_index]);
        }
        __syncthreads();

        // scale current column and update trailing matrix
        if(myrow > k)
        {
            rA[k] *= pivot_value;
            for(I j = k + 1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];
        }
        __syncthreads();
    }

    // write results to global memory
    if(myrow < DIM)
        ipiv[myrow] = mypiv + offset;
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        A[myrow + j * lda] = rA[j];
}

/** getf2_npvt_small_kernel (non pivoting version) **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_npvt_small_kernel(const I m,
                            U AA,
                            const rocblas_stride shiftA,
                            const I lda,
                            const rocblas_stride strideA,
                            INFO* infoA,
                            const I batch_count,
                            const I offset)
{
    using S = decltype(std::real(T{}));

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = reinterpret_cast<T*>(lmem);
    T* val = common + hipBlockDim_y * DIM;
    common += ty * DIM;

    // local variables
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j * lda];

        // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        // share pivot row
        if(myrow == k)
        {
            val[ty] = rA[k];
            for(I j = k + 1; j < DIM; ++j)
                common[j] = rA[j];

            if(val[ty] != T(0))
                val[ty] = S(1) / val[ty];
        }
        __syncthreads();

        // check singularity
        if(val[ty] == 0 && myinfo == 0)
            myinfo = k + 1;

        // scale current column and update trailing matrix
        if(myrow > k)
        {
            rA[k] *= val[ty];
            for(I j = k + 1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];
        }
        __syncthreads();
    }

    // write results to global memory
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        A[myrow + j * lda] = rA[j];
}

/** getf2_panel_kernel takes care of small matrices with m >= n **/
template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void getf2_panel_kernel(const I m,
                                         const I n,
                                         U AA,
                                         const rocblas_stride shiftA,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         I* ipivA,
                                         const rocblas_stride shiftP,
                                         const rocblas_stride strideP,
                                         INFO* infoA,
                                         const I batch_count,
                                         const I offset,
                                         I* permut_idx,
                                         const rocblas_stride stridePI)
{
    using S = decltype(std::real(T{}));

    const I tx = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_z;
    const I bdx = hipBlockDim_x;
    const I bdy = hipBlockDim_y;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    I* ipiv = load_ptr_batch<I>(ipivA, id, shiftP, strideP);
    I* permut = (permut_idx != nullptr ? permut_idx + id * stridePI : nullptr);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    extern __shared__ double lmem[];
    T* x = reinterpret_cast<T*>(lmem);
    T* y = x + bdx;
    S* sval = reinterpret_cast<S*>(y + n);
    I* sidx = reinterpret_cast<I*>(sval + bdx);
    __shared__ T val;

    // local variables
    S val1, val2;
    T valtmp, pivot_val;
    I idx1, idx2, pivot_idx;
    INFO myinfo = 0; // to build info

    // init step: read column zero from A
    if(ty == 0)
    {
        valtmp = (tx < m) ? A[tx] : 0;
        idx1 = tx;
        x[tx] = valtmp;
        val1 = aabs<S>(valtmp);
        sval[tx] = val1;
        sidx[tx] = idx1;
    }

    // main loop (for each pivot)
    for(I k = 0; k < n; ++k)
    {
        // find pivot (maximum in column)
        __syncthreads();
        for(I i = bdx / 2; i > 0; i /= 2)
        {
            if(tx < i && ty == 0)
            {
                val2 = sval[tx + i];
                idx2 = sidx[tx + i];
                if((val1 < val2) || (val1 == val2 && idx1 > idx2))
                {
                    sval[tx] = val1 = val2;
                    sidx[tx] = idx1 = idx2;
                }
            }
            __syncthreads();
        }
        pivot_idx = sidx[0]; //after reduction this is the index of max value
        pivot_val = x[pivot_idx];

        // check singularity and scale value for current column
        if(pivot_val == T(0))
        {
            pivot_idx = k;
            if(myinfo == 0)
                myinfo = k + 1;
        }
        else
            pivot_val = S(1) / pivot_val;

        // update ipiv
        if(tx == 0 && ty == 0)
            ipiv[k] = pivot_idx + 1 + offset;

        // update column k
        if(tx != pivot_idx)
        {
            pivot_val *= x[tx];
            if(ty == 0 && tx >= k && tx < m)
                A[tx + k * lda] = pivot_val;
        }

        // put pivot row in shared mem
        if(tx < n && ty == 0)
        {
            y[tx] = A[pivot_idx + tx * lda];
            if(tx == k)
                val = pivot_val;
        }
        __syncthreads();

        // swap pivot row with updated row k
        if(tx < n && ty == 0 && pivot_idx != k)
        {
            valtmp = (tx == k) ? val : A[k + tx * lda];
            valtmp -= (tx > k) ? val * y[tx] : 0;
            A[pivot_idx + tx * lda] = valtmp;
            A[k + tx * lda] = y[tx];
            if(tx == k + 1)
            {
                x[pivot_idx] = valtmp;
                val1 = aabs<S>(valtmp);
                sval[pivot_idx] = val1;
            }
            if(permut_idx && tx == k)
                swap(permut[k], permut[pivot_idx]);
        }

        // complete the rank update
        if(tx > k && tx < m && tx != pivot_idx)
        {
            for(I j = ty + k + 2; j < n; j += bdy)
            {
                valtmp = A[tx + j * lda];
                valtmp -= pivot_val * y[j];
                A[tx + j * lda] = valtmp;
            }

            if(ty == 0 && k < n - 1)
            {
                valtmp = A[tx + (k + 1) * lda];
                valtmp -= pivot_val * y[k + 1];
                A[tx + (k + 1) * lda] = valtmp;
                x[tx] = valtmp;
                val1 = aabs<S>(valtmp);
                sval[tx] = val1;
            }
        }

        // update ipiv and prepare for next step
        if(tx <= k && ty == 0)
        {
            val1 = 0;
            x[tx] = 0;
            sval[tx] = 0;
        }
        idx1 = tx;
        if(ty == 0)
            sidx[tx] = idx1;
    }

    // update info
    if(tx == 0 && *info == 0 && myinfo > 0 && ty == 0)
        *info = myinfo + offset;
}

/** getf2_npvt_panel_kernel (non pivoting version) **/
template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void getf2_npvt_panel_kernel(const I m,
                                              const I n,
                                              U AA,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              INFO* infoA,
                                              const I batch_count,
                                              const I offset)
{
    using S = decltype(std::real(T{}));

    const I tx = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_z;
    const I bdx = hipBlockDim_x;
    const I bdy = hipBlockDim_y;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    extern __shared__ double lmem[];
    T* x = reinterpret_cast<T*>(lmem);
    T* y = x + bdx;
    __shared__ T val;

    // local variables
    T pivot_val, val1;
    INFO myinfo = 0; // to build info

    // init step: read column zero from A
    if(ty == 0)
    {
        val1 = (tx < m) ? A[tx] : 0;
        x[tx] = val1;
    }

    // main loop (for each pivot)
    for(I k = 0; k < n; ++k)
    {
        __syncthreads();
        pivot_val = x[k];

        // check singularity and scale value for current column
        if(pivot_val == T(0) && myinfo == 0)
            myinfo = k + 1;
        else
            pivot_val = S(1) / pivot_val;

        // update column k
        if(tx != k)
        {
            pivot_val *= x[tx];
            if(ty == 0 && tx >= k && tx < m)
                A[tx + k * lda] = pivot_val;
        }

        // put pivot row in shared mem
        if(tx < n && ty == 0)
        {
            y[tx] = A[k + tx * lda];
            if(tx == k)
                val = pivot_val;
        }
        __syncthreads();

        // complete the rank update
        if(tx > k && tx < m)
        {
            for(I j = ty + k + 2; j < n; j += bdy)
            {
                val1 = A[tx + j * lda];
                val1 -= pivot_val * y[j];
                A[tx + j * lda] = val1;
            }

            if(ty == 0 && k < n - 1)
            {
                val1 = A[tx + (k + 1) * lda];
                val1 -= pivot_val * y[k + 1];
                A[tx + (k + 1) * lda] = val1;
                x[tx] = val1;
            }
        }

        // prepare for next step
        if(tx <= k && ty == 0)
            x[tx] = 0;
    }

    // update info
    if(tx == 0 && *info == 0 && myinfo > 0 && ty == 0)
        *info = myinfo + offset;
}

/** getf2_scale_update_kernel executes an optimized scaled rank-update (scal + ger)
    for panel matrices (matrices with less than 128 columns).
    Useful to speedup the factorization of block-columns in getrf **/
template <typename T, typename I, typename U>
//template <rocblas_int N, typename T, typename U>
ROCSOLVER_KERNEL void getf2_scale_update_kernel(const I m,
                                                const I n,
                                                T* pivotval,
                                                U AA,
                                                const rocblas_stride shiftA,
                                                const I lda,
                                                const rocblas_stride strideA)
{
    // indices
    I bid = hipBlockIdx_z;
    I tx = hipThreadIdx_x;
    I ty = hipThreadIdx_y;
    I i = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + tx;

    // shared data arrays
    T pivot, val;
    extern __shared__ double lmem[];
    T* x = reinterpret_cast<T*>(lmem);
    T* y = x + hipBlockDim_x;

    // batch instance
    T* A = load_ptr_batch(AA, bid, shiftA + 1 + lda, strideA);
    T* X = load_ptr_batch(AA, bid, shiftA + 1, strideA);
    T* Y = load_ptr_batch(AA, bid, shiftA + lda, strideA);
    pivot = pivotval[bid];

    // read data from global to shared memory
    I j = tx * hipBlockDim_y + ty;
    if(j < n)
        y[j] = Y[j * lda];

    // scale
    if(ty == 0 && i < m)
    {
        x[tx] = X[i];
        x[tx] *= pivot;
        X[i] = x[tx];
    }
    __syncthreads();

    // rank update; put computed values back to global memory
    if(i < m)
    {
#pragma unroll
        for(I j = ty; j < n; j += hipBlockDim_y)
        {
            val = A[i + j * lda];
            val -= x[tx] * y[j];
            A[i + j * lda] = val;
        }
    }
}

/*************************************************************
    Launchers of specilized  kernels
*************************************************************/

/** launcher of getf2_small_kernel **/
template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_small(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride)
{
#define RUN_LUFACT_SMALL(DIM)                                                                      \
    if(pivot)                                                                                      \
        ROCSOLVER_LAUNCH_KERNEL((getf2_small_kernel<DIM, T>), grid, block, lmemsize, stream, m, A, \
                                shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count,    \
                                offset, permut_idx, stride);                                       \
    else                                                                                           \
        ROCSOLVER_LAUNCH_KERNEL((getf2_npvt_small_kernel<DIM, T>), grid, block, lmemsize, stream,  \
                                m, A, shiftA, lda, strideA, info, batch_count, offset)

    // determine sizes
    I opval[] = {GETF2_OPTIM_NGRP};
    I ngrp = (batch_count < 2 || m > 32) ? 1 : opval[m - 1];
    I blocks = (batch_count - 1) / ngrp + 1;
    I nthds = m;
    I msize;
    if(pivot)
        msize = std::max(m, n);
    else
        msize = n + 1;

    // prepare kernel launch
    dim3 grid(1, blocks, 1);
    dim3 block(nthds, ngrp, 1);
    size_t lmemsize = msize * ngrp * sizeof(T);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    // kernel launch
    switch(n)
    {
    case 1: RUN_LUFACT_SMALL(1); break;
    case 2: RUN_LUFACT_SMALL(2); break;
    case 3: RUN_LUFACT_SMALL(3); break;
    case 4: RUN_LUFACT_SMALL(4); break;
    case 5: RUN_LUFACT_SMALL(5); break;
    case 6: RUN_LUFACT_SMALL(6); break;
    case 7: RUN_LUFACT_SMALL(7); break;
    case 8: RUN_LUFACT_SMALL(8); break;
    case 9: RUN_LUFACT_SMALL(9); break;
    case 10: RUN_LUFACT_SMALL(10); break;
    case 11: RUN_LUFACT_SMALL(11); break;
    case 12: RUN_LUFACT_SMALL(12); break;
    case 13: RUN_LUFACT_SMALL(13); break;
    case 14: RUN_LUFACT_SMALL(14); break;
    case 15: RUN_LUFACT_SMALL(15); break;
    case 16: RUN_LUFACT_SMALL(16); break;
    case 17: RUN_LUFACT_SMALL(17); break;
    case 18: RUN_LUFACT_SMALL(18); break;
    case 19: RUN_LUFACT_SMALL(19); break;
    case 20: RUN_LUFACT_SMALL(20); break;
    case 21: RUN_LUFACT_SMALL(21); break;
    case 22: RUN_LUFACT_SMALL(22); break;
    case 23: RUN_LUFACT_SMALL(23); break;
    case 24: RUN_LUFACT_SMALL(24); break;
    case 25: RUN_LUFACT_SMALL(25); break;
    case 26: RUN_LUFACT_SMALL(26); break;
    case 27: RUN_LUFACT_SMALL(27); break;
    case 28: RUN_LUFACT_SMALL(28); break;
    case 29: RUN_LUFACT_SMALL(29); break;
    case 30: RUN_LUFACT_SMALL(30); break;
    case 31: RUN_LUFACT_SMALL(31); break;
    case 32: RUN_LUFACT_SMALL(32); break;
    case 33: RUN_LUFACT_SMALL(33); break;
    case 34: RUN_LUFACT_SMALL(34); break;
    case 35: RUN_LUFACT_SMALL(35); break;
    case 36: RUN_LUFACT_SMALL(36); break;
    case 37: RUN_LUFACT_SMALL(37); break;
    case 38: RUN_LUFACT_SMALL(38); break;
    case 39: RUN_LUFACT_SMALL(39); break;
    case 40: RUN_LUFACT_SMALL(40); break;
    case 41: RUN_LUFACT_SMALL(41); break;
    case 42: RUN_LUFACT_SMALL(42); break;
    case 43: RUN_LUFACT_SMALL(43); break;
    case 44: RUN_LUFACT_SMALL(44); break;
    case 45: RUN_LUFACT_SMALL(45); break;
    case 46: RUN_LUFACT_SMALL(46); break;
    case 47: RUN_LUFACT_SMALL(47); break;
    case 48: RUN_LUFACT_SMALL(48); break;
    case 49: RUN_LUFACT_SMALL(49); break;
    case 50: RUN_LUFACT_SMALL(50); break;
    case 51: RUN_LUFACT_SMALL(51); break;
    case 52: RUN_LUFACT_SMALL(52); break;
    case 53: RUN_LUFACT_SMALL(53); break;
    case 54: RUN_LUFACT_SMALL(54); break;
    case 55: RUN_LUFACT_SMALL(55); break;
    case 56: RUN_LUFACT_SMALL(56); break;
    case 57: RUN_LUFACT_SMALL(57); break;
    case 58: RUN_LUFACT_SMALL(58); break;
    case 59: RUN_LUFACT_SMALL(59); break;
    case 60: RUN_LUFACT_SMALL(60); break;
    case 61: RUN_LUFACT_SMALL(61); break;
    case 62: RUN_LUFACT_SMALL(62); break;
    case 63: RUN_LUFACT_SMALL(63); break;
    case 64: RUN_LUFACT_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }

    return rocblas_status_success;
}

/** launcher of getf2_panel_kernel **/
template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_panel(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride)
{
    using S = decltype(std::real(T{}));

    // determine sizes
    I dimy, dimx;
    if(m <= 8)
        dimx = 8;
    else if(m <= 16)
        dimx = 16;
    else if(m <= 32)
        dimx = 32;
    else if(m <= 64)
        dimx = 64;
    else if(m <= 128)
        dimx = 128;
    else if(m <= 256)
        dimx = 256;
    else if(m <= 512)
        dimx = 512;
    else
        dimx = 1024;
    dimy = I(1024) / dimx;

    // prepare kernel launch
    dim3 grid(1, 1, batch_count);
    dim3 block(dimx, dimy, 1);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(pivot)
    {
        size_t lmemsize = (dimx + n) * sizeof(T) + dimx * (sizeof(I) + sizeof(S));
        ROCSOLVER_LAUNCH_KERNEL((getf2_panel_kernel<T>), grid, block, lmemsize, stream, m, n, A,
                                shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count,
                                offset, permut_idx, stride);
    }
    else
    {
        size_t lmemsize = (dimx + n) * sizeof(T);
        ROCSOLVER_LAUNCH_KERNEL((getf2_npvt_panel_kernel<T>), grid, block, lmemsize, stream, m, n,
                                A, shiftA, lda, strideA, info, batch_count, offset);
    }

    return rocblas_status_success;
}

/** launcher of getf2_scale_update_kernel **/
template <typename T, typename I, typename U>
void getf2_run_scale_update(rocblas_handle handle,
                            const I m,
                            const I n,
                            T* pivotval,
                            U A,
                            const rocblas_stride shiftA,
                            const I lda,
                            const rocblas_stride strideA,
                            const I batch_count,
                            const I dimx,
                            const I dimy)
{
    size_t lmemsize = sizeof(T) * (dimx + n);
    I blocks = (m - 1) / dimx + 1;
    dim3 threads(dimx, dimy, 1);
    dim3 grid(blocks, 1, batch_count);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // scale and update trailing matrix with local function
    ROCSOLVER_LAUNCH_KERNEL((getf2_scale_update_kernel<T>), grid, threads, lmemsize, stream, m, n,
                            pivotval, A, shiftA, lda, strideA);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GETF2_SMALL(T, I, INFO, U)                                           \
    template rocblas_status getf2_run_small<T, I, INFO, U>(                              \
        rocblas_handle handle, const I m, const I n, U A, const rocblas_stride shiftA,   \
        const I lda, const rocblas_stride strideA, I* ipiv, const rocblas_stride shiftP, \
        const rocblas_stride strideP, INFO* info, const I batch_count, const bool pivot, \
        const I offset, I* permut_idx, const rocblas_stride stride)
#define INSTANTIATE_GETF2_PANEL(T, I, INFO, U)                                           \
    template rocblas_status getf2_run_panel<T, I, INFO, U>(                              \
        rocblas_handle handle, const I m, const I n, U A, const rocblas_stride shiftA,   \
        const I lda, const rocblas_stride strideA, I* ipiv, const rocblas_stride shiftP, \
        const rocblas_stride strideP, INFO* info, const I batch_count, const bool pivot, \
        const I offset, I* permut_idx, const rocblas_stride stride)
#define INSTANTIATE_GETF2_SCALE_UPDATE(T, I, U)                                                  \
    template void getf2_run_scale_update<T, I, U>(rocblas_handle handle, const I m, const I n,   \
                                                  T* pivotval, U A, const rocblas_stride shiftA, \
                                                  const I lda, const rocblas_stride strideA,     \
                                                  const I batch_count, const I dimx, const I dimy)

ROCSOLVER_END_NAMESPACE

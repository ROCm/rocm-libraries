/************************************************************************
 * Small kernel algorithm based on:
 * Abdelfattah, A., Haidar, A., Tomov, S., & Dongarra, J. (2017).
 * Factorization and inversion of a million matrices using GPUs: Challenges
 * and countermeasures. Procedia Computer Science, 108, 606-615.
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "asan_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

/** getf2_small_kernel takes care of of matrices with m < n
    m <= GETF2_MAX_THDS and n <= GETF2_MAX_COLS.
    DIM is the capacity of the per-thread row array and must be >= n; the number
    of columns n is a run-time argument, so one instantiation serves every n <= DIM.
    The loops run to DIM and skip the columns past n: the bound must be a
    compile-time constant for every index into rA to be fixed after unrolling,
    which keeps rA in registers instead of scratch memory. **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_small_kernel(const I m,
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

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_x * static_cast<I>(hipBlockDim_y) + ty;

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
    common += ty * std::max(m, n);

    // local variables
    T pivot_value;
    T test_value;
    I pivot_index;
    I mypiv = myrow + 1; // to build ipiv
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values
    const I kmax = std::min(m, n);

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        if(j < n)
            rA[j] = A[myrow + j * lda];

            // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        if(k < kmax)
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

            // synchronize across waves before overwriting common
            __syncthreads();

            // swap rows (lazy swaping)
            if(myrow == pivot_index)
            {
                myrow = k;
                // share pivot row
                for(I j = k + 1; j < DIM; ++j)
                    if(j < n)
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
                    if(j < n)
                        rA[j] -= rA[k] * common[j];
            }
            __syncthreads();
        }
    }

    // write results to global memory
    if(myrow < n)
        ipiv[myrow] = mypiv + offset;
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        if(j < n)
            A[myrow + j * lda] = rA[j];
}

/** getf2_npvt_small_kernel (non pivoting version). DIM and n as in
    getf2_small_kernel. **/
template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_npvt_small_kernel(const I m,
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

    I myrow = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I id = hipBlockIdx_x * static_cast<I>(hipBlockDim_y) + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    INFO* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = reinterpret_cast<T*>(lmem);
    T* val = common + hipBlockDim_y * n;
    common += ty * n;

    // local variables
    INFO myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values
    const I kmax = std::min(m, n);

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        if(j < n)
            rA[j] = A[myrow + j * lda];

            // for each pivot (main loop)
#pragma unroll DIM
    for(I k = 0; k < DIM; ++k)
    {
        if(k < kmax)
        {
            // share pivot row
            if(myrow == k)
            {
                val[ty] = rA[k];
                for(I j = k + 1; j < DIM; ++j)
                    if(j < n)
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
                    if(j < n)
                        rA[j] -= rA[k] * common[j];
            }
            __syncthreads();
        }
    }

    // write results to global memory
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
#pragma unroll DIM
    for(I j = 0; j < DIM; ++j)
        if(j < n)
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
    // batch blocks go on grid.x (limit 2^31), not grid.y: grid.y is capped at
    // 65536 on some archs (e.g. gfx1201), which overflows for large batches.
    dim3 grid(blocks, 1, 1);
    dim3 block(nthds, ngrp, 1);
    size_t lmemsize = msize * ngrp * sizeof(T);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // n is passed at run time; DIM only sizes the per-thread row array. The sizes step by 8,
    // so the array is never much larger than n and the register use stays close to what n
    // needs. A row of 64 double-complex values does not fit in registers, so double complex
    // also gets a size of 60, the largest that does; for other types that slot is 64.
    ROCSOLVER_ASSUME_X(n >= 1 && n <= GETF2_SSKER_MAX_N, "getf2_run_small needs small n");
    constexpr bool is_z = std::is_same_v<T, rocblas_double_complex>;
    constexpr int DIM60 = is_z ? 60 : 64;
    const int sz = (n <= 56) ? (n - 1) / 8 : (is_z && n > 60) ? 8 : 7;
    if(pivot)
    {
        auto kernel = std::array{
            getf2_small_kernel<8, T, I, INFO, U>,  getf2_small_kernel<16, T, I, INFO, U>,
            getf2_small_kernel<24, T, I, INFO, U>, getf2_small_kernel<32, T, I, INFO, U>,
            getf2_small_kernel<40, T, I, INFO, U>, getf2_small_kernel<48, T, I, INFO, U>,
            getf2_small_kernel<56, T, I, INFO, U>, getf2_small_kernel<DIM60, T, I, INFO, U>,
            getf2_small_kernel<64, T, I, INFO, U>};
        ROCSOLVER_LAUNCH_KERNEL(kernel[sz], grid, block, lmemsize, stream, m, n, A, shiftA, lda,
                                strideA, ipiv, shiftP, strideP, info, batch_count, offset,
                                permut_idx, stride);
    }
    else
    {
        auto kernel = std::array{getf2_npvt_small_kernel<8, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<16, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<24, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<32, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<40, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<48, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<56, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<DIM60, T, I, INFO, U>,
                                 getf2_npvt_small_kernel<64, T, I, INFO, U>};
        ROCSOLVER_LAUNCH_KERNEL(kernel[sz], grid, block, lmemsize, stream, m, n, A, shiftA, lda,
                                strideA, info, batch_count, offset);
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
    constexpr I max_threads = ROCSOLVER_ASAN_VALUE(256, 1024);
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
    else if constexpr(!rocsolver_enable_asan)
    {
        if(m <= 256)
            dimx = 256;
        else if(m <= 512)
            dimx = 512;
        else
            dimx = 1024;
    }
    else
        dimx = 256;
    dimy = I(max_threads) / dimx;

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

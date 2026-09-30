/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @brief Optimized TRSV batched kernel that uses z-dimension of thread blocks
 * 
 * This kernel is designed for cases where batch_count >> N (by a factor of 8+).
 * By using the z-dimension of thread blocks, multiple batches can be processed
 * concurrently within the same thread block.
 * 
 * Key optimization: When batch_count >= 8*N, each thread block processes multiple
 * batches using threadIdx.z
 */

// Block solve for lower triangular with z-dimension batching
template <rocblas_int BLOCK_LDA, rocblas_int DIM_Z, bool UNIT, typename T>
void ROCBLAS_KERNEL_ILF rocblas_trsv_block_solve_lower_big_batch(const T* __restrict__ A,
                                                                 T* xshared,
                                                                 T& val)
{
    const int tz = threadIdx.z;

    // Iterate forwards through the diagonal block
    for(int i = 0; i < BLOCK_LDA; i++)
    {
        // Solve current element (only thread with x-index matching current row)
        if(threadIdx.x == i && threadIdx.y == 0)
        {
            if(!UNIT)
                val *= A[i * BLOCK_LDA + i]; // Multiply by diagonal element
            xshared[tz] = val; // Store solved value in shared memory
        }

        __syncthreads();

        // Update future elements with solved one
        if(threadIdx.x > i && threadIdx.y == 0)
        {
            val += A[i * BLOCK_LDA + threadIdx.x] * xshared[tz];
        }

        __syncthreads();
    }
}

// Block solve for upper triangular with z-dimension batching
template <rocblas_int BLOCK_LDA, rocblas_int DIM_Z, bool UNIT, typename T>
void ROCBLAS_KERNEL_ILF rocblas_trsv_block_solve_upper_big_batch(const T* __restrict__ A,
                                                                 T* xshared,
                                                                 T& val)
{
    const int tz = threadIdx.z;

    // Iterate backwards through the diagonal block
    for(int i = BLOCK_LDA - 1; i >= 0; i--)
    {
        // Solve current element
        if(threadIdx.x == i && threadIdx.y == 0)
        {
            if(!UNIT)
                val *= A[i * BLOCK_LDA + i];
            xshared[tz] = val;
        }

        __syncthreads();

        // Update future elements with solved one
        if(threadIdx.x < i && threadIdx.y == 0)
        {
            val += A[i * BLOCK_LDA + threadIdx.x] * xshared[tz];
        }

        __syncthreads();
    }
}

/**
 * @brief Main TRSV kernel using z-dimension for batch parallelization
 * 
 * @tparam DIM_X     Block size for the matrix (typically 16 or 32)
 * @tparam DIM_Y     Number of threads per row (typically 16)
 * @tparam DIM_Z     Number of batches per thread block (typically 8)
 * @tparam LOWER     True for lower triangular, false for upper
 * @tparam TRANS     True if transposed
 * @tparam CONJ      True if conjugate transpose
 * @tparam UNIT      True if unit diagonal
 * @tparam T         Data type (float, double, complex)
 */
template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int DIM_Z,
          bool        LOWER,
          bool        TRANS,
          bool        CONJ,
          bool        UNIT,
          typename T,
          typename ALPHATYPE,
          typename ATYPE,
          typename XTYPE>
ROCBLAS_KERNEL(DIM_X* DIM_Y* DIM_Z)
rocblas_trsv_big_batch_device(rocblas_int    n,
                              ATYPE          dA,
                              rocblas_stride offset_A,
                              int64_t        lda,
                              rocblas_stride stride_A,
                              ALPHATYPE      alpha_device_host,
                              XTYPE          dx,
                              rocblas_stride offset_x,
                              int64_t        incx,
                              rocblas_stride stride_x,
                              rocblas_int*   w_completed_sec,
                              rocblas_int    batch_count)
{
    // Determine if we need forward or backward substitution
    constexpr bool backwards_sub = (!LOWER && !TRANS) || (LOWER && TRANS);

    const rocblas_int num_blocks = gridDim.y;
    const rocblas_int tx         = threadIdx.x;
    const rocblas_int ty         = threadIdx.y;
    const rocblas_int tz         = threadIdx.z;

    // This allows processing DIM_Z batches per thread block
    uint32_t batch = blockIdx.x * DIM_Z + threadIdx.z;

    T alpha = load_scalar(alpha_device_host);

    // Shared memory arrays dimensioned for DIM_Z batches
    // This allows each batch (z-index) to have its own workspace
    __shared__ T sum_z[DIM_X * DIM_Y * DIM_Z];
    __shared__ T sAdiag_z[DIM_X * DIM_X * DIM_Z];
    __shared__ T sx_z[DIM_X * DIM_Z];

    // Shared memory per batch in z-dimension
    __shared__ T xs[DIM_Z];

    // Register storage for off-diagonal block
    // keeping DIM_X == DIM_Y for big_batch so DIM_X / DIM_Y = 1
    T sAoff[DIM_X / DIM_Y];

    if(batch < batch_count)
    {

        // Load pointers for this specific batch
        auto* __restrict__ A = load_ptr_batch(dA, batch, offset_A, stride_A);
        auto* __restrict__ x = load_ptr_batch(dx, batch, offset_x, stride_x);

        // Offset into shared memory for this batch's z-slice
        int xyz_offset = (DIM_X * DIM_Y) * tz;
        int sx_offset  = DIM_X * tz;
        T*  sum        = sum_z + xyz_offset;
        T*  sAdiag     = sAdiag_z + xyz_offset;
        T*  sx         = sx_z + sx_offset;

        const int tid = DIM_X * ty + tx;

        // Assign to register row in each thread
        rocblas_int block_row = backwards_sub ? num_blocks - 1 - blockIdx.y : blockIdx.y;

        // If problem is not divisible into DIM_X sized sections, the last block row
        // will be smaller and must be handled differently
        const rocblas_int remainder        = n % DIM_X;
        const bool        row_is_remainder = ((n - 1) / DIM_X == block_row && remainder != 0);

        // Store square block of A beside triangular part (if not first row)
        const bool first_row = backwards_sub ? block_row == num_blocks - 1 : block_row == 0;
        if(!first_row)
        {
            const rocblas_int block_col = backwards_sub ? block_row + 1 : block_row - 1;
            const rocblas_int local_col = TRANS ? block_row * DIM_X + tx : block_col * DIM_X + ty;
            const rocblas_int local_row = TRANS ? block_col * DIM_X + ty : block_row * DIM_X + tx;
            const size_t      A_idx     = (local_row) + (local_col)*lda;

            // for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
            {
                // const size_t i_idx = TRANS ? i : i * lda;

                __syncthreads();
                if(TRANS ? (local_row /* + i */ < n && local_col < n)
                         : (local_row < n && local_col /* + i */ < n))
                    sAoff[0] = A[A_idx /* + i_idx */]; // 0 = i / DIM_Y
                else
                    sAoff[0] = 0.0; // i / DIM_Y
            }
        }

        // Storing diagonal block of A into shared memory for subtitution solve
#ifdef INV_AFTER
        bool cache_transpose = (TRANS && LOWER && num_blocks - 1 - block_row < INV_AFTER)
                               || (TRANS && !LOWER && block_row < INV_AFTER)
                               || (TRANS && row_is_remainder);
#else
        bool cache_transpose = TRANS; // works for ALL without inversion method
#endif
        {
            rocblas_int       row    = tx;
            const rocblas_int col    = ty; // + i;
            const rocblas_int sA_idx = cache_transpose ? col + DIM_X * row : col * DIM_X + row;
            const size_t A_idx = (block_row * DIM_X * lda + block_row * DIM_X) + col * lda + row;

            if(!row_is_remainder)
            {
                // commented out code for loop i < DIM_X allows juxtaposition of anisotropic tile sizes support in default kernel
                // i.e. DIM_X != DIM_Y which we don't support here

                //rocblas_int row = tx;
                //for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
                {
                    // const rocblas_int col    = ty; // + i;
                    // const rocblas_int sA_idx = cache_transpose ? col + DIM_X * row : col * DIM_X + row;
                    // const size_t      A_idx
                    //     = (block_row * DIM_X * lda + block_row * DIM_X) + col * lda + row;
                    // const rocblas_int total_col = block_row * DIM_X + col;
                    // const rocblas_int total_row = block_row * DIM_X + row;

                    if((row > col && LOWER) || (col > row && !LOWER))
                    {
                        sAdiag[sA_idx] = CONJ ? -conj(A[A_idx]) : -A[A_idx];
                    }
                    else if(!UNIT && row == col)
                    {
                        // Dividing here so we can just multiply later.
                        sAdiag[sA_idx] = 1.0 / (CONJ ? conj(A[A_idx]) : A[A_idx]);
                    }
                    else if(col < DIM_X && row < DIM_X) // In off-triangular portion - set to 0
                    {
                        sAdiag[sA_idx] = 0.0;
                    }
                }
            }
            else // remainder of a block
            {
                //rocblas_int row = tx;
                //for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
                {
                    // const rocblas_int col    = ty; //ty + i;
                    // const rocblas_int sA_idx = cache_transpose ? col + DIM_X * row : col * DIM_X + row;
                    // const size_t      A_idx
                    //     = (block_row * DIM_X * lda + block_row * DIM_X) + col * lda + row;
                    // const rocblas_int total_col = block_row * DIM_X + col;
                    // const rocblas_int total_row = block_row * DIM_X + row;
                    if(((row > col && LOWER) || (col > row && !LOWER)) && row < remainder
                       && col < remainder)
                    {
                        sAdiag[sA_idx] = CONJ ? -conj(A[A_idx]) : -A[A_idx];
                    }
                    else if(!UNIT && row == col && row < remainder)
                    {
                        // Dividing here so we can just multiply later.
                        sAdiag[sA_idx] = 1.0 / (CONJ ? conj(A[A_idx]) : A[A_idx]);
                    }
                    else if(col < DIM_X
                            && row < DIM_X) // In off-triangular portion or past end of remainder
                    {
                        sAdiag[sA_idx] = 0.0;
                    }
                }
            }
        }
        __syncthreads();

#ifdef INV_AFTER
        if(((block_row >= INV_AFTER && !backwards_sub)
            || (num_blocks - 1 - block_row >= INV_AFTER && backwards_sub))
           && !row_is_remainder)
        {
            if(LOWER)
                rocblas_trsv_invert<T, DIM_X, DIM_X, DIM_X, DIM_Y, UNIT, TRANS>(sAdiag, sum);
            else
                rocblas_trsv_invert_upper<T, DIM_X, DIM_X, DIM_X, DIM_Y, UNIT, TRANS>(sAdiag, sum);
        }
#endif
        __syncthreads();

        // Store relevant x value into register
        T val = 0;
        if(ty == 0)
        {
            if(!row_is_remainder || tx < remainder)
            {
                // multiply by alpha when reading from device memory x
                val = -alpha * x[(block_row * DIM_X + tx) * incx];
            }
        }

        // Once previously solved block is ready, apply this to other square blocks
        rocblas_int       col_done = -1;
        const rocblas_int iters    = backwards_sub ? num_blocks - 1 - block_row : block_row;
        for(rocblas_int block_iter = 0; block_iter < iters; block_iter++)
        {
            // For backwards substitution, we start at the bottom and propogate upwards, else we go top-to-bottom
            const rocblas_int block_col
                = backwards_sub ? (num_blocks - 1 - block_iter) : block_iter;

            const rocblas_int local_col = TRANS ? block_row * DIM_X + tx : block_col * DIM_X + ty;
            const rocblas_int local_row = TRANS ? block_col * DIM_X + ty : block_row * DIM_X + tx;
            const size_t      A_idx     = local_col * lda + local_row;
            const int64_t     x_idx     = (block_col * DIM_X) * incx;

            if(tid == 0)
            {
                // Wait until the previous column is done. Use global memory to
                // update when ready.
                if(col_done < block_iter)
                {
                    while(w_completed_sec[batch] < block_iter)
                        __threadfence();
                    col_done = w_completed_sec[batch];
                }
            }

            // Few intermittent failures without this. Needed to wait for updated x values, I guess?
            __threadfence();
            __syncthreads();

            // Store x val (of previous block) into shared memory
            if(tid < DIM_X)
            {
                if(block_col * DIM_X + tid >= n)
                    sx[tid] = 0.0;
                else
                {
                    // Don't multiply by alpha here as this is a solved value
                    sx[tid] = x[x_idx + tid * incx];
                }
            }

            __syncthreads();

            // Update val with result of previous block
            //for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
            {
                // Use shared memory if previous col since we cached this earlier
                //const size_t i_idx = TRANS ? i : i * lda;
                const bool cached
                    = !first_row
                      && (backwards_sub ? block_col == block_row + 1 : block_col == block_row - 1);

                if(TRANS ? (local_row /* + i */ < n && local_col < n)
                         : (local_row < n && local_col /* + i */ < n))
                {
                    auto A_val = cached ? sAoff[0] : A[A_idx /* + i_idx */]; // i / DIM_Y
                    if(CONJ)
                        A_val = conj(A_val);
                    val += A_val * sx[/* i */ +ty];
                }
            }
        }

        // Add "solved" x values into shared memory to be summed further
        sum[ty * DIM_X + tx] = val;
        __syncthreads();

        if(ty == 0)
        {
            // Sum DIM_Y elements into single val
            for(rocblas_int i = 1; i < DIM_Y; i++)
            {
                val += sum[i * DIM_X + tx];
            }
            val = -val;

            if(row_is_remainder && tx >= remainder)
                val = 0.0; // zero out out-of-bounds
        }

        // Solve the current block.
        // It's important that we're very efficient here, as other blocks are
        // likely just waiting for the result of this block.
#ifdef INV_AFTER
        if(((block_row >= INV_AFTER && !backwards_sub)
            || (num_blocks - 1 - block_row >= INV_AFTER && backwards_sub))
           && !row_is_remainder)
        {
            rocblas_trsv_block_solve_inverse<T, DIM_X, DIM_Y, backwards_sub>(sAdiag, sx, val, sum);

            if(!row_is_remainder || tx < remainder)
            {
                if(ty == 0)
                {
                    x[(block_row * DIM_X + tid) * incx] = val;
                }
            }
        }
        else // same as without inversion
        {
            // Solve the diagonal block
            if constexpr(backwards_sub)
                rocblas_trsv_block_solve_upper_big_batch<DIM_X, DIM_Z, UNIT>(sAdiag, xs, val);
            else
                rocblas_trsv_block_solve_lower_big_batch<DIM_X, DIM_Z, UNIT>(sAdiag, xs, val);

            // Store solved value into x
            if(!row_is_remainder || tx < remainder)
                if(ty == 0)
                    x[(block_row * DIM_X + tid) * incx] = val;
        }
#else
        // Solve the diagonal block
        if constexpr(backwards_sub)
            rocblas_trsv_block_solve_upper_big_batch<DIM_X, DIM_Z, UNIT>(sAdiag, xs, val);
        else
            rocblas_trsv_block_solve_lower_big_batch<DIM_X, DIM_Z, UNIT>(sAdiag, xs, val);

        // Store solved value into x
        if(!row_is_remainder || tx < remainder)
            if(ty == 0)
                x[(block_row * DIM_X + tid) * incx] = val;
#endif

        // ensure solved x values are saved
        __threadfence();

        // next column is ready
        // don't need an atomic op here since there should only
        // be one block for each batch here at once
        __syncthreads(); // for windows instability
        if(tid == 0)
            w_completed_sec[batch]++;

        __threadfence();

    } // if
}

ROCBLAS_KERNEL(1024) rocblas_trsv_init_big_batch(int batch_count, rocblas_int* w_completed_sec)
{
    // The last block section which has been completed (for each batch)
    int batch = blockIdx.x * 1024 + threadIdx.x;
    if(batch < batch_count)
        w_completed_sec[batch] = -1;
}

/**
 * @brief Wrapper function to launch the large batch count TRSV kernel
 * 
 * This function should be called when batch_count >= 8*n to take advantage
 * of the z-dimension parallelization.
 */
template <rocblas_int DIM_X, typename T, typename ATYPE, typename XTYPE>
rocblas_status rocblas_internal_trsv_substitution_big_batch_template(rocblas_handle    handle,
                                                                     rocblas_fill      uplo,
                                                                     rocblas_operation transA,
                                                                     rocblas_diagonal  diag,
                                                                     rocblas_int       n,
                                                                     ATYPE             dA,
                                                                     rocblas_stride    offset_A,
                                                                     int64_t           lda,
                                                                     rocblas_stride    stride_A,
                                                                     const T*          alpha,
                                                                     XTYPE             dx,
                                                                     rocblas_stride    offset_x,
                                                                     int64_t           incx,
                                                                     rocblas_stride    stride_x,
                                                                     rocblas_int       batch_count,
                                                                     rocblas_int* w_completed_sec)
{
    if(!n || !batch_count)
        return rocblas_status_success;

    // Initialize completion tracking
    ROCBLAS_LAUNCH_KERNEL(rocblas_trsv_init_big_batch,
                          dim3((batch_count - 1) / 1024 + 1),
                          dim3(1024),
                          0,
                          handle->get_stream(),
                          batch_count,
                          w_completed_sec);

    offset_x = incx < 0 ? offset_x + incx * (1 - n) : offset_x;

    // Use z-dimension for batch parallelization
    constexpr rocblas_int DIM_Y = 4;
    static_assert(DIM_X == DIM_Y, "Square sub blocks");
    constexpr rocblas_int DIM_Z = 64; // Process 64 batches per thread block

    rocblas_int blocks      = (n - 1) / DIM_X + 1;
    rocblas_int batch_grids = (batch_count - 1) / DIM_Z + 1;

    dim3 threads(DIM_X, DIM_Y, DIM_Z);
    dim3 grid(batch_grids, blocks);

    bool alpha_exists = false;
    T    alpha_local  = 1.0;
    if(alpha != nullptr)
    {
        alpha_exists = true;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
            alpha_local = *alpha;
    }

#define TRSV_TEMPLATE_PARAMS(alpha_)                                                              \
    grid, threads, 0, handle->get_stream(), n, dA, offset_A, lda, stride_A, alpha_, dx, offset_x, \
        incx, stride_x, w_completed_sec, batch_count

    if(handle->pointer_mode == rocblas_pointer_mode_device && alpha_exists)
    {
        // Template Parameters: DIM_X, DIM_Y, DIM_Z, LOWER, TRANSPOSE, CONJUGATE, UNIT_DIAG, T
        if(uplo == rocblas_fill_upper)
        {
            if(diag == rocblas_diagonal_unit)
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_conjugate_transpose)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                }
            }
            else
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
            }
        }
        else
        {
            if(diag == rocblas_diagonal_unit)
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
            }
            else
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha));
            }
        }
    }
    else
    {
        // Template Parameters: DIM_X, DIM_Y, DIM_Z, LOWER, TRANSPOSE, CONJUGATE, UNIT_DIAG, T
        if(uplo == rocblas_fill_upper)
        {
            if(diag == rocblas_diagonal_unit)
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_conjugate_transpose)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                }
            }
            else
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         false,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
            }
        }
        else
        {
            if(diag == rocblas_diagonal_unit)
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
            }
            else
            {
                if(transA == rocblas_operation_none)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
                else if(transA == rocblas_operation_conjugate_transpose)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_trsv_big_batch_device<DIM_X,
                                                                         DIM_Y,
                                                                         DIM_Z,
                                                                         true,
                                                                         true,
                                                                         true,
                                                                         false,
                                                                         T>),
                                          TRSV_TEMPLATE_PARAMS(alpha_local));
            }
        }
    }

#undef TRSV_TEMPLATE_PARAMS

    return rocblas_status_success;
}

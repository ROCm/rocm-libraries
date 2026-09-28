/* **************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_host_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_gemm_device_functions.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

#include <climits>

ROCSOLVER_BEGIN_NAMESPACE

/** GEMM device function to compute C = alpha * A * B + beta * C.

    Call this kernel with 'batch_count' groups in z, and enough
    groups in x and y to cover all the 'm' rows and 'n' columns of C. **/
template <typename T, typename I, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void gemm_kernel(const I m,
                                  const I n,
                                  const I k,
                                  V alpha,
                                  bool const conjA,
                                  U1 AA,
                                  rocblas_stride const shiftA,
                                  I const inca,
                                  I const lda,
                                  rocblas_stride const strideA,
                                  bool const conjB,
                                  U2 BB,
                                  rocblas_stride const shiftB,
                                  I const incb,
                                  I const ldb,
                                  rocblas_stride const strideB,
                                  V beta,
                                  U3 CC,
                                  rocblas_stride const shiftC,
                                  I const incc,
                                  I const ldc,
                                  rocblas_stride const strideC,
                                  I const batch_count)
{
    // batch indices

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    I const j_start = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + hipThreadIdx_y;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // batch instance
        T const a = load_scalar(alpha, bid, 0);
        T const b = load_scalar(beta, bid, 0);
        T const* const A = load_ptr_batch(AA, bid, shiftA, strideA);
        T const* const B = load_ptr_batch(BB, bid, shiftB, strideB);
        T* const C = load_ptr_batch(CC, bid, shiftC, strideC);

        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i < m; i += i_inc)
            {
                // gemm function
                T temp = 0;
                for(I idx = 0; idx < k; idx++)
                {
                    const auto Aval = conjA ? conj(A[i * inca + idx * lda]) : A[i * inca + idx * lda];
                    const auto Bval = conjB ? conj(B[idx * incb + j * ldb]) : B[idx * incb + j * ldb];
                    temp += Aval * Bval;
                }
                C[i * incc + j * ldc] = a * temp + b * C[i * incc + j * ldc];
            }
        }

    } // end for bid
}

#if ROCSOLVER_MFMA_ENABLED

/** GEMM device function to compute C = alpha * A * B + beta * C.

    Call this kernel with 'batch_count' groups in z. Each wave in x
    and y computes 16 of 'm' rows and 16 of the 'n' columns of C. **/
template <typename T, typename I, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void mfma_gemm_kernel(rocblas_operation transA,
                                       rocblas_operation transB,
                                       const I m,
                                       const I n,
                                       const I p,
                                       V alpha,
                                       U1 AA,
                                       rocblas_stride shiftA,
                                       I inca,
                                       I lda,
                                       rocblas_stride strideA,
                                       U2 BB,
                                       rocblas_stride shiftB,
                                       I incb,
                                       I ldb,
                                       rocblas_stride strideB,
                                       V beta,
                                       U3 CC,
                                       rocblas_stride shiftC,
                                       I incc,
                                       I ldc,
                                       rocblas_stride strideC,
                                       I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    for(I batch_id = bid_start; batch_id < batch_count; batch_id += bid_inc)
    {
        const I bid_x = blockIdx.x;
        const I bid_y = blockIdx.y;

        const I numWaves_x = blockDim.x / warpSize;
        const I numWaves_y = blockDim.y;

        const I wid_x = threadIdx.x / warpSize;
        const I wid_y = threadIdx.y;

        const I block_row = 16 * (numWaves_x * bid_x + wid_x);
        const I block_col = 16 * (numWaves_y * bid_y + wid_y);

        if(block_row >= m || block_col >= n)
        {
            return;
        }

        const I m_bar = (block_row + 16) <= m ? 16 : m % 16;
        const I n_bar = (block_col + 16) <= n ? 16 : n % 16;

        // batch instance
        T a = load_scalar(alpha, batch_id, 0);
        T b = load_scalar(beta, batch_id, 0);
        T* A = load_ptr_batch(AA, batch_id, shiftA, strideA);
        T* B = load_ptr_batch(BB, batch_id, shiftB, strideB);
        T* C = load_ptr_batch(CC, batch_id, shiftC, strideC);

        A += block_row * (transA == rocblas_operation_none ? inca : lda);
        B += block_col * (transB == rocblas_operation_none ? ldb : incb);

        // C(bid_x,bid_y) += A(bid_x,:) * B(:,bid_y)
        gemm_16x16xp(transA, transB, m_bar, n_bar, p, a, A, inca, lda, B, incb, ldb, b,
                     C + (block_col * ldc + block_row * incc), incc, ldc);
    } // end for batch_id
}

#else // ROCSOLVER_MFMA_ENABLED
template <typename T, typename I, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void mfma_gemm_kernel(rocblas_operation transA,
                                       rocblas_operation transB,
                                       const I m,
                                       const I n,
                                       const I p,
                                       V alpha,
                                       U1 AA,
                                       rocblas_stride shiftA,
                                       I inca,
                                       I lda,
                                       rocblas_stride strideA,
                                       U2 BB,
                                       rocblas_stride shiftB,
                                       I incb,
                                       I ldb,
                                       rocblas_stride strideB,
                                       V beta,
                                       U3 CC,
                                       rocblas_stride shiftC,
                                       I incc,
                                       I ldc,
                                       rocblas_stride strideC,
                                       I const batch_count)
{
}
#endif // ROCSOLVER_MFMA_ENABLED

/*************************************************************
    Launchers of specialized kernels
*************************************************************/

template <typename T, typename I, typename U1, typename U2, typename U3>
ROCSOLVER_EXPORT rocblas_status rocsolver_gemm(rocblas_handle handle,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               I m,
                                               I n,
                                               I k,
                                               const T* alpha,
                                               U1 A,
                                               rocblas_stride shiftA,
                                               I inca,
                                               I lda,
                                               rocblas_stride strideA,
                                               U2 B,
                                               rocblas_stride shiftB,
                                               I incb,
                                               I ldb,
                                               rocblas_stride strideB,
                                               const T* beta,
                                               U3 C,
                                               rocblas_stride shiftC,
                                               I incc,
                                               I ldc,
                                               rocblas_stride strideC,
                                               I batch_count,
                                               T** work)
{
    ROCSOLVER_ENTER("gemm", "transA:", transA, "transB:", transB, "m:", m, "n:", n, "k:", k,
                    "shiftA:", shiftA, "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb,
                    "ldb:", ldb, "shiftC:", shiftC, "incc:", incc, "ldc:", ldc, "bc:", batch_count);

    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

#ifndef USE_INTERNAL_GEMM
    if(inca == 1 && incb == 1 && incc == 1)
        return rocblasCall_gemm(handle, transA, transB, m, n, k, alpha, A, shiftA, lda, strideA, B,
                                shiftB, ldb, strideB, beta, C, shiftC, ldc, strideC, batch_count,
                                work);
#endif

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode pmode;
    rocblas_get_pointer_mode(handle, &pmode);

    // get warp size
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    std::string deviceArch(props->gcnArchName);

    if((deviceArch.find("gfx90a") != std::string::npos)
       || (deviceArch.find("gfx940") != std::string::npos)
       || (deviceArch.find("gfx941") != std::string::npos)
       || (deviceArch.find("gfx942") != std::string::npos))
    {
        const auto warpSize = props->warpSize;

        // launch specialized kernel
        const I numWarpsX = 4;
        const I numWarpsY = 4;
        const I blocksx = (m + (numWarpsX * 16 - 1)) / (numWarpsX * 16);
        const I blocksy = (n + (numWarpsY * 16 - 1)) / (numWarpsY * 16);

        I const max_blocks = get_nblocks_yz(handle);

        // ---------------------------------
        // TODO: handle  the case blocksy > 64 * 1024
        // ---------------------------------
        dim3 grid(blocksx, blocksy, std::min(max_blocks, batch_count));
        dim3 threads(numWarpsX * warpSize, numWarpsY, 1);
        if(pmode == rocblas_pointer_mode_device)
        {
            ROCSOLVER_LAUNCH_KERNEL((mfma_gemm_kernel<T>), grid, threads, 0, stream, transA, transB,
                                    m, n, k, alpha, A, shiftA, inca, lda, strideA, B, shiftB, incb,
                                    ldb, strideB, beta, C, shiftC, incc, ldc, strideC, batch_count);
        }
        else
        {
            ROCSOLVER_LAUNCH_KERNEL((mfma_gemm_kernel<T>), grid, threads, 0, stream, transA, transB,
                                    m, n, k, *alpha, A, shiftA, inca, lda, strideA, B, shiftB, incb,
                                    ldb, strideB, *beta, C, shiftC, incc, ldc, strideC, batch_count);
        }
    }
    else
    {
        // matrices can be transposed by swapping inc and ld
        I lda1 = inca;
        I lda2 = lda;
        I ldb1 = incb;
        I ldb2 = ldb;
        if(transA != rocblas_operation_none)
        {
            lda1 = lda;
            lda2 = inca;
        }
        if(transB != rocblas_operation_none)
        {
            ldb1 = ldb;
            ldb2 = incb;
        }

        const bool conjA = transA == rocblas_operation_conjugate_transpose;
        const bool conjB = transB == rocblas_operation_conjugate_transpose;

        // launch specialized kernel
        I blocksx = (m - 1) / BS2 + 1;
        I blocksy = (n - 1) / BS2 + 1;

        I const max_blocks = get_nblocks_yz(handle);

        dim3 grid(blocksx, std::min(max_blocks, blocksy), std::min(max_blocks, batch_count));
        dim3 threads(BS2, BS2, 1);
        if(pmode == rocblas_pointer_mode_device)
        {
            ROCSOLVER_LAUNCH_KERNEL((gemm_kernel<T>), grid, threads, 0, stream, m, n, k, alpha,
                                    conjA, A, shiftA, lda1, lda2, strideA, conjB, B, shiftB, ldb1,
                                    ldb2, strideB, beta, C, shiftC, incc, ldc, strideC, batch_count);
        }
        else
        {
            ROCSOLVER_LAUNCH_KERNEL((gemm_kernel<T>), grid, threads, 0, stream, m, n, k, *alpha,
                                    conjA, A, shiftA, lda1, lda2, strideA, conjB, B, shiftB, ldb1,
                                    ldb2, strideB, *beta, C, shiftC, incc, ldc, strideC, batch_count);
        }
    }

    return rocblas_status_success;
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <typename T, typename I, typename U1, typename U2, typename U3>
ROCSOLVER_EXPORT inline rocblas_status rocsolver_gemm(rocblas_handle handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      I m,
                                                      I n,
                                                      I k,
                                                      const T* alpha,
                                                      U1 A,
                                                      rocblas_stride shiftA,
                                                      I lda,
                                                      rocblas_stride strideA,
                                                      U2 B,
                                                      rocblas_stride shiftB,
                                                      I ldb,
                                                      rocblas_stride strideB,
                                                      const T* beta,
                                                      U3 C,
                                                      rocblas_stride shiftC,
                                                      I ldc,
                                                      rocblas_stride strideC,
                                                      I batch_count,
                                                      T** work)
{
    return rocsolver_gemm<T, I>(handle, transA, transB, m, n, k, alpha, A, shiftA, 1, lda, strideA,
                                B, shiftB, 1, ldb, strideB, beta, C, shiftC, 1, ldc, strideC,
                                batch_count, work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GEMM(T, I, U1, U2, U3)                                                        \
    template ROCSOLVER_EXPORT rocblas_status rocsolver_gemm<T, I, U1, U2, U3>(                    \
        rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, I m, I n, I k, \
        const T* alpha, U1 A, rocblas_stride shiftA, I lda, rocblas_stride strideA, U2 B,         \
        rocblas_stride shiftB, I ldb, rocblas_stride strideB, const T* beta, U3 C,                \
        rocblas_stride shiftC, I ldc, rocblas_stride strideC, I batch_count, T** work)

ROCSOLVER_END_NAMESPACE

/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
                                  bool conjA,
                                  U1 AA,
                                  rocblas_stride shiftA,
                                  I inca,
                                  I lda,
                                  rocblas_stride strideA,
                                  bool conjB,
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
                                  rocblas_stride strideC)
{
    // indices
    I bid = hipBlockIdx_z;
    I i = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I j = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + hipThreadIdx_y;

    // batch instance
    T a = load_scalar(alpha, bid, 0);
    T b = load_scalar(beta, bid, 0);
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* B = load_ptr_batch(BB, bid, shiftB, strideB);
    T* C = load_ptr_batch(CC, bid, shiftC, strideC);

    // gemm function assuming no conjugation
    T temp = 0;
    if(i < m && j < n)
    {
        for(I idx = 0; idx < k; idx++)
        {
            const auto Aval = conjA ? conj(A[i * inca + idx * lda]) : A[i * inca + idx * lda];
            const auto Bval = conjB ? conj(B[idx * incb + j * ldb]) : B[idx * incb + j * ldb];
            temp += Aval * Bval;
        }
        C[i * incc + j * ldc] = a * temp + b * C[i * incc + j * ldc];
    }
}

// /** Optimized kernel that executes a simple gemm A = BC
//     where A, B and C are sub blocks of the same matrix MM with
//     leading dimension ldim and stride. A, B and C are
//     located in MM by their respective shifts.

//     Call this kernel with 'batch_count' groups in z, and enough
//     groups in x and y to cover all the 'm' rows and 'n' columns of C.
//     Size of shared memory per group should be:
//     lmemsize = k * (hipBlockDim_x + hipBlockDim_y) * sizeof(T); **/
// template <typename T, typename U>
// ROCSOLVER_KERNEL void gemm_kernel(const rocblas_int m,
//                                   const rocblas_int n,
//                                   const rocblas_int k,
//                                   U MM,
//                                   const rocblas_int shiftA,
//                                   const rocblas_int shiftB,
//                                   const rocblas_int shiftC,
//                                   const rocblas_int ldim,
//                                   const rocblas_stride stride)
// {
//     // indices
//     int id = hipBlockIdx_z;
//     int tx = hipThreadIdx_x;
//     int ty = hipThreadIdx_y;
//     int bdx = hipBlockDim_x;
//     int bdy = hipBlockDim_y;
//     int i = hipBlockIdx_x * bdx + tx;
//     int j = hipBlockIdx_y * bdy + ty;

//     // batch instance
//     T* A = load_ptr_batch(MM, id, shiftA, stride);
//     T* B = load_ptr_batch(MM, id, shiftB, stride);
//     T* C = load_ptr_batch(MM, id, shiftC, stride);

//     // shared mem setup
//     extern __shared__ double lmem[];
//     T* a = reinterpret_cast<T*>(lmem);
//     T* b = a + k * bdx;
//     T c;

//     // local row and column of the shared arrays
//     a += tx * k;
//     b += ty * k;

//     // read A and B into shared mem
//     for(int kk = ty; kk < k; kk += bdy)
//         a[kk] = i < m ? A[i + kk * ldim] : 0;
//     for(int kk = tx; kk < k; kk += bdx)
//         b[kk] = j < n ? B[kk + j * ldim] : 0;
//     __syncthreads();

//     if(i < m && j < n)
//     {
//         // update c
//         c = C[i + j * ldim];
//         for(int kk = 0; kk < k; ++kk)
//             c -= a[kk] * b[kk];

//         // write back to global memory
//         C[i + j * ldim] = c;
//     }
// }

/*************************************************************
    Launchers of specialized kernels
*************************************************************/

template <typename T, typename I, typename U1, typename U2, typename U3>
rocblas_status rocsolver_gemm(rocblas_handle handle,
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
    dim3 grid(blocksx, blocksy, batch_count);
    dim3 threads(BS2, BS2, 1);
    if(pmode == rocblas_pointer_mode_device)
    {
        ROCSOLVER_LAUNCH_KERNEL((gemm_kernel<T>), grid, threads, 0, stream, m, n, k, alpha, conjA,
                                A, shiftA, lda1, lda2, strideA, conjB, B, shiftB, ldb1, ldb2,
                                strideB, beta, C, shiftC, incc, ldc, strideC);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((gemm_kernel<T>), grid, threads, 0, stream, m, n, k, *alpha, conjA,
                                A, shiftA, lda1, lda2, strideA, conjB, B, shiftB, ldb1, ldb2,
                                strideB, *beta, C, shiftC, incc, ldc, strideC);
    }

    return rocblas_status_success;
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <typename T, typename I, typename U1, typename U2, typename U3>
inline rocblas_status rocsolver_gemm(rocblas_handle handle,
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
    template rocblas_status rocsolver_gemm<T, I, U1, U2, U3>(                                     \
        rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, I m, I n, I k, \
        const T* alpha, U1 A, rocblas_stride shiftA, I lda, rocblas_stride strideA, U2 B,         \
        rocblas_stride shiftB, I ldb, rocblas_stride strideB, const T* beta, U3 C,                \
        rocblas_stride shiftC, I ldc, rocblas_stride strideC, I batch_count, T** work)

ROCSOLVER_END_NAMESPACE

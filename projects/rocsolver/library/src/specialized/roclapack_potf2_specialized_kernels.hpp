/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver_run_specialized_kernels.hpp"
#include <algorithm>
#include <cmath>

ROCSOLVER_BEGIN_NAMESPACE

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block per matrix.
 * ------------------------------------------------------
 *
 * NB           Number of panels to perform blocked decomposition.
 *              ceildiv(n, PANEL_SIZE)
 * PANEL_SIZE   Size of panel to perform non-blocked decomposition.
 *              PANEL_SIZE == BlockDim.x == BlockDim.y
**/
template <int NB, int PANEL_SIZE, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void potf2_kernel_small(const bool is_upper,
                                         const I n,
                                         U AA,
                                         const rocblas_stride shiftA,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         INFO* const info)
{
    auto const tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    auto const inc = hipBlockDim_y * hipBlockDim_x;
    auto const tidx = hipThreadIdx_x;
    auto const tidy = hipThreadIdx_y;

    assert(hipBlockDim_z == 1);

    // get batch index
    auto const bid = hipBlockIdx_z;
    assert(AA != nullptr);
    assert(info != nullptr);

    T* const A = load_ptr_batch(AA, bid, shiftA, strideA);
    INFO* const info_bid = info + bid;

    extern __shared__ rocblas_int lsmem[];
    T* Ash = reinterpret_cast<T*>(lsmem);
    auto constexpr ldash = NB * PANEL_SIZE;

    bool failed = false;

    /* -----------------------------------------------------------
    Load PANEL_SIZE x PANEL_SIZE blocks of A into registers. Where
    each thread in a workgroup has a corresponding element in the
    matrix block.
    If uplo == rocblas_fill_upper then only the upper triangular
    blocks are stored, otherwise only the lower triangular blocks
    are stored.
    ----------------------------------------------------------- */
    T Arg[(NB * (NB + 1)) / 2] = {0};

    I arg_idx = 0;
    for(I jb = 0; jb < NB; jb++)
    {
        for(I i = jb; i < NB; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto row = jb * PANEL_SIZE + tidx;
                if(col < n && row < n && row <= col)
                {
                    const auto idx = col * lda + row;
                    Arg[arg_idx] = A[idx];
                }
            }
            else
            {
                const auto col = jb * PANEL_SIZE + tidy;
                const auto row = i * PANEL_SIZE + tidx;
                if(col < n && row < n && row >= col)
                {
                    const auto idx = col * lda + row;
                    Arg[arg_idx] = A[idx];
                }
            }

            arg_idx++;
        }
    }

    /* ---------------------------------------------------------
    Panel Cholesky decomposition: iterate through each panel and
    decompose in LDS. Update the trailing matrix in register.
    --------------------------------------------------------- */
    arg_idx = 0;
    for(I kb = 0; kb < NB; kb++)
    {
        /* --------------------------------------------------------
        Write panel A(kb:kb+PANEL_SIZE, kb:N)^T to LDS if upper,
        or A(kb:N, kb:kb+PANEL_SIZE) otherwise. (N = NB*PANEL_SIZE)
        -------------------------------------------------------- */
        for(I i = 0; i < NB - kb; i++)
        {
            // write to lds as lower and compute as lower
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto idx = tidx * ldash + col;
                Ash[idx] = Arg[arg_idx + i];
            }
            else
            {
                const auto row = i * PANEL_SIZE + tidx;
                const auto idx = tidy * ldash + row;
                Ash[idx] = Arg[arg_idx + i];
            }
        }

        __syncthreads();

        I nn = n - kb * PANEL_SIZE;

        /* ----------------------------
        Factor panel matrix Ash in LDS.
        ---------------------------- */
        for(I kcol = 0; kcol < PANEL_SIZE; kcol++)
        {
            if(kcol >= nn)
                break;

            auto kk = kcol * ldash + kcol;
            auto const akk = std::real(Ash[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));

            __syncthreads();

            if(!isok)
            {
                if(tid == 0)
                {
                    Ash[kk] = akk;
                    // Fortran 1-based index
                    if(*info_bid == 0)
                        *info_bid = kb * PANEL_SIZE + kcol + 1;
                }
                failed = true;
                __syncthreads();
                break;
            }

            // -----------------------------------------------------
            //   (1) |l11|^2 = a11 =>  l11 = sqrt(a11), get diagonal
            // -----------------------------------------------------

            auto const lkk = std::sqrt(akk);
            if(tid == 0)
            {
                Ash[kk] = lkk;
            }

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const conj_lkk = conj(lkk);
            for(I j0 = (kcol + 1) + tid; j0 < nn; j0 += inc)
            {
                auto const j0k = j0 + kcol * ldash;

                Ash[j0k] = (Ash[j0k] / conj_lkk);
            }

            __syncthreads();

            // --------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21(0:PANEL_SIZE)', update trailing
            //
            //   note: only update elements below the diagonal
            // --------------------------------------------------------------

            for(I j = (kcol + 1) + tidy; j < PANEL_SIZE; j += hipBlockDim_y)
            {
                auto const vj = Ash[j + kcol * ldash];
                for(I i = j + tidx; i < nn; i += hipBlockDim_x)
                {
                    auto const vi = Ash[i + kcol * ldash];
                    auto const ij = i + j * ldash;

                    Ash[ij] = Ash[ij] - vi * conj(vj);
                }
            }
            __syncthreads();
        }

        /* ------------------------------------
        Update the trailing matrix in register.
        ------------------------------------ */
        I upd_arg_idx = arg_idx + NB - kb;
        for(I j = kb + 1; j < NB; j++)
        {
            for(I i = j; i < NB; i++)
            {
                if(is_upper)
                {
                    // ---------------------------------------------------------------------------------
                    // formulate gemm problem:
                    //    A22 = A(kb+PANEL_SIZE:N, kb+PANEL_SIZE:N)
                    //    A12 = Ash(PANEL_SIZE:N, 0:PANEL_SIZE)^T = A(kb:kb+PANEL_SIZE, kb+PANEL_SIZE:N)
                    //
                    //    operation: A22 = A12' * A12
                    // ---------------------------------------------------------------------------------
                    const auto col = (i - kb) * PANEL_SIZE + tidy;
                    const auto row = (j - kb) * PANEL_SIZE + tidx;

                    for(I p = 0; p < PANEL_SIZE; p++)
                    {
                        Arg[upd_arg_idx + i - j] -= conj(Ash[row + p * ldash]) * Ash[col + p * ldash];
                    }
                }
                else
                {
                    // -------------------------------------------------------------------------------
                    // formulate gemm problem:
                    //    A22 = A(kb+PANEL_SIZE:N, kb+PANEL_SIZE:N)
                    //    A21 = Ash(PANEL_SIZE:N, 0:PANEL_SIZE) = A(kb+PANEL_SIZE:N, kb:kb+PANEL_SIZE)
                    //
                    //    operation: A22 = A21 * A21'
                    // -------------------------------------------------------------------------------
                    const auto col = (j - kb) * PANEL_SIZE + tidy;
                    const auto row = (i - kb) * PANEL_SIZE + tidx;

                    for(I p = 0; p < PANEL_SIZE; p++)
                    {
                        Arg[upd_arg_idx + i - j] -= Ash[row + p * ldash] * conj(Ash[col + p * ldash]);
                    }
                }
            }
            upd_arg_idx += NB - j;
        }

        /* ---------------------------------
        Load panel in LDS back to registers.
        --------------------------------- */
        for(I i = 0; i < NB - kb; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto idx = tidx * ldash + col;
                Arg[arg_idx + i] = Ash[idx];
            }
            else
            {
                const auto row = i * PANEL_SIZE + tidx;
                const auto idx = tidy * ldash + row;
                Arg[arg_idx + i] = Ash[idx];
            }
        }
        arg_idx += NB - kb;

        __syncthreads();

        if(failed)
            break;
    }

    /* ----------------------------------------------------
    Write blocks of the matrix in registers back to memory.
    ---------------------------------------------------- */
    arg_idx = 0;
    for(I jb = 0; jb < NB; jb++)
    {
        for(I i = jb; i < NB; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto row = jb * PANEL_SIZE + tidx;
                if(col < n && row < n && row <= col)
                {
                    const auto idx = col * lda + row;
                    A[idx] = Arg[arg_idx];
                }
            }
            else
            {
                const auto col = jb * PANEL_SIZE + tidy;
                const auto row = i * PANEL_SIZE + tidx;
                if(col < n && row < n && row >= col)
                {
                    const auto idx = col * lda + row;
                    A[idx] = Arg[arg_idx];
                }
            }

            arg_idx++;
        }
    }
}

/*************************************************************
    Launchers of specilized kernels
*************************************************************/

template <typename T, typename I, typename INFO, typename U>
rocblas_status potf2_run_small(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               INFO* info,
                               const I batch_count)
{
    ROCSOLVER_ENTER("potf2_kernel_small", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const auto nb = (n + BS2 - 1) / BS2;

    size_t lmemsize = sizeof(T) * nb * BS2 * BS2;

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    if(lmemsize > props->sharedMemPerBlock)
    {
        return rocblas_status_internal_error;
    }

    bool const is_upper = (uplo == rocblas_fill_upper);
    auto kernel = std::array{
        potf2_kernel_small<1, BS2, T, I, INFO, U>, potf2_kernel_small<2, BS2, T, I, INFO, U>,
        potf2_kernel_small<3, BS2, T, I, INFO, U>, potf2_kernel_small<4, BS2, T, I, INFO, U>,
        potf2_kernel_small<5, BS2, T, I, INFO, U>, potf2_kernel_small<6, BS2, T, I, INFO, U>,
        potf2_kernel_small<7, BS2, T, I, INFO, U>, potf2_kernel_small<8, BS2, T, I, INFO, U>};
    ROCSOLVER_LAUNCH_KERNEL(kernel[nb - 1], dim3(1, 1, batch_count), dim3(BS2, BS2, 1), lmemsize,
                            stream, is_upper, n, A, shiftA, lda, strideA, info);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_POTF2_SMALL(T, I, INFO, U)                                                       \
    template rocblas_status potf2_run_small<T, I, INFO, U>(                                          \
        rocblas_handle handle, const rocblas_fill uplo, const I n, U A, const rocblas_stride shiftA, \
        const I lda, const rocblas_stride strideA, INFO* info, const I batch_count)

ROCSOLVER_END_NAMESPACE

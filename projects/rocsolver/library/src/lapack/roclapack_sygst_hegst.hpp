/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "roclapack_sygs2_hegs2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

static bool constexpr use_sygs2_hegs2_alt = true;

// ----------------------------------------------------------
// kernel to symmetrize matrix and
// copy the strictly lower or strictly upper triangular part
// ----------------------------------------------------------
template <typename T, typename I, typename U>
static __global__ void copy_symm_tri_kernel(bool const is_lower,
                                            I const n,

                                            U AA,
                                            rocblas_stride const shiftA,
                                            I const lda,
                                            rocblas_stride const strideA,

                                            T* const BB,
                                            rocblas_stride const strideB,

                                            I const batch_count,
                                            bool const is_restore)
{
    // ---------------------------------------
    // linear mapping from strictly upper/lower
    // part to array of size n*(n-1)/2
    // ---------------------------------------
    auto idxU = [=](auto i, auto j) {
        assert(i < j);
        auto const k = ((i * n - i * (i + 1) / 2) + (j - (i + 1)));
        assert((0 <= k) && (k < n * (n - 1) / 2));
        return (k);
    };
    auto idxL = [=](auto i, auto j) {
        assert(i > j);
        auto const k = idxU(j, i);
        assert((0 <= k) && (k < n * (n - 1) / 2));
        return (k);
    };

    auto idx2D = [=](auto i, auto j, auto ld) { return (i + j * ld); };

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    rocblas_stride const shiftB = 0;
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        T* const B = load_ptr_batch<T>(BB, bid, shiftB, strideB);

        for(I j = 0 + j_start; j < n; j += j_inc)
        {
            I const row_start = (is_lower) ? j + 1 : 0;
            I const row_end = (is_lower) ? n : j;

            for(I i = row_start + i_start; i < row_end; i += i_inc)
            {
                auto const ij_B = (is_lower) ? idxU(j, i) : idxL(j, i);

                if(is_restore)
                {
                    auto const ji_A = idx2D(j, i, lda);
                    A[ji_A] = B[ij_B];
                }
                else
                {
                    // -------------------
                    // save other triangular part of A to B
                    // and symmetrize A
                    // -------------------
                    auto const ij_A = idx2D(i, j, lda);
                    auto const ji_A = idx2D(j, i, lda);

                    auto const aij = A[ij_A];
                    auto const aji = A[ji_A];
                    A[ji_A] = conj(aij);
                    B[ij_B] = aji;
                }
            } // end for i
        } // end for j
    } // end for bid
}

template <typename T, typename I, typename U>
static void copy_symm_tri(rocblas_handle handle,
                          bool const is_lower,
                          I const n,

                          U AA,
                          rocblas_stride const shiftA,
                          I const lda,
                          rocblas_stride const strideA,

                          T* const BB,
                          rocblas_stride const strideB,

                          I const batch_count,
                          bool const is_restore)
{
    auto ceildiv = [](auto m, auto b) { return ((m + b - 1) / b); };

    I const nx = 64;
    I const ny = 4;
    I const max_blocks = 1024;
    I const nbx = std::min(max_blocks, ceildiv(n, nx));
    I const nby = std::min(max_blocks, ceildiv(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    copy_symm_tri_kernel<T, I, U><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(

        is_lower, n,

        AA, shiftA, lda, strideA,

        BB, strideB,

        batch_count, is_restore);
}
template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_sygst_hegst_getMemorySize(const rocblas_fill uplo,
                                         const rocblas_eform itype,
                                         const I n,
                                         const I batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work_x_temp,
                                         size_t* size_workArr_temp_arr,
                                         size_t* size_store_wcs_invA,
                                         size_t* size_invA_arr,
                                         bool* optim_mem)
{
    // if quick return no need of workspace
    *size_scalars = 0;
    *size_work_x_temp = 0;
    *size_workArr_temp_arr = 0;
    *size_store_wcs_invA = 0;
    *size_invA_arr = 0;
    *optim_mem = true;
    if(n == 0 || batch_count == 0)
    {
        return;
    }

    if(n < xxGST_BLOCKSIZE)
    {
        // requirements for calling a single SYGS2/HEGS2
        rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, n, batch_count, size_scalars,
                                                        size_work_x_temp, size_store_wcs_invA,
                                                        size_workArr_temp_arr);
        *size_invA_arr = 0;
        *optim_mem = true;
    }
    else
    {
        I kb = xxGST_BLOCKSIZE;
        size_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

        // requirements for calling SYGS2/HEGS2 for the subblocks
        rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, kb, batch_count, size_scalars,
                                                        size_work_x_temp, size_store_wcs_invA,
                                                        size_workArr_temp_arr);
        *size_invA_arr = 0;

        if(itype == rocblas_eform_ax)
        {
            // extra requirements for calling TRSM
            if(uplo == rocblas_fill_upper)
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, n - kb, kb,
                    batch_count, &temp1, &temp2, &temp3, &temp4, optim_mem);
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none,
                                                        n - kb, kb, batch_count, &temp5, &temp6,
                                                        &temp7, &temp8, optim_mem);
            }
            else
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none,
                                                        n - kb, kb, batch_count, &temp1, &temp2,
                                                        &temp3, &temp4, optim_mem);
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n - kb, kb,
                    batch_count, &temp5, &temp6, &temp7, &temp8, optim_mem);
            }

            *size_work_x_temp = std::max(*size_work_x_temp, std::max(temp1, temp5));
            *size_workArr_temp_arr = std::max(*size_workArr_temp_arr, std::max(temp2, temp6));
            *size_store_wcs_invA = std::max(*size_store_wcs_invA, std::max(temp3, temp7));
            *size_invA_arr = std::max(*size_invA_arr, std::max(temp4, temp8));
        }
        else
            *optim_mem = true;
    }

    if(use_sygs2_hegs2_alt)
    {
        // -----------------------------------
        // extra requirements for calling TRSM
        // -----------------------------------

        I const nn = xxGST_BLOCKSIZE;
        size_t temp1{}, temp2{}, temp3{}, temp4{}, temp5{}, temp6{}, temp7{}, temp8{};

        if(itype == rocblas_eform_ax)
        {
            if(uplo == rocblas_fill_upper)
            {
                // -------------------------
                // Compute A <- inv(U')*A*inv(U)
                // -------------------------
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, nn, nn, batch_count,
                    &temp1, &temp2, &temp3, &temp4, optim_mem);

                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none,
                                                        nn, nn, batch_count, &temp5, &temp6, &temp7,
                                                        &temp8, optim_mem);
            }
            else
            {
                // -------------------------
                // Compute A <- inv(L)*A*inv(L')
                // -------------------------
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none,
                                                        nn, nn, batch_count, &temp1, &temp2, &temp3,
                                                        &temp4, optim_mem);
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, nn, nn, batch_count,
                    &temp5, &temp6, &temp7, &temp8, optim_mem);
            }

            *size_work_x_temp = std::max(*size_work_x_temp, std::max(temp1, temp5));
            *size_workArr_temp_arr = std::max(*size_workArr_temp_arr, std::max(temp2, temp6));
            *size_store_wcs_invA = std::max(*size_store_wcs_invA, std::max(temp3, temp7));
            *size_invA_arr = std::max(*size_invA_arr, std::max(temp4, temp8));
        }

        // ----------------------------------------
        // storage to save strictly triangular part
        //
        // expand storage for work_x_temp
        //
        // NOTE: assume xxGST_BLOCKSIZE is a power of 2
        // to maintain alignment
        // ----------------------------------------
        size_t const size_Asave = (sizeof(T) * nn * (nn - 1) / 2) * batch_count;
        *size_work_x_temp += size_Asave;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, typename I>
rocblas_status rocsolver_sygst_hegst_template(rocblas_handle handle,
                                              const rocblas_eform itype,
                                              const rocblas_fill uplo,
                                              const I n,
                                              U A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              U B,
                                              const rocblas_stride shiftB,
                                              const I ldb,
                                              const rocblas_stride strideB,
                                              const I batch_count,
                                              T* scalars,
                                              void* work_x_temp,
                                              void* workArr_temp_arr,
                                              void* store_wcs_invA,
                                              void* invA_arr,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("sygst_hegst", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I nb = xxGST_BLOCKSIZE;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    S s_one = 1;
    T t_one = 1;
    T t_half = 0.5;
    T t_minone = -1;
    T t_minhalf = -0.5;

    auto sygs2_hegs2_alt
        = [](rocblas_handle handle, rocblas_eform const itype, rocblas_fill const uplo, I const n,

             auto A, rocblas_stride const shiftA, I const lda, rocblas_stride const strideA,

             auto B, rocblas_stride const shiftB, I const ldb, rocblas_stride const strideB,

             I const batch_count, T* const Asave,

             bool const optim_mem, auto temp1, auto temp2, auto temp3, auto temp4) -> rocblas_status {
        // ------------------------------------------------------------------
        // Let B = R * R',   B = L * L', => R = L,   or B = U' * U => R = U'
        //
        // rocblas_eform_ax     A * x = lambda * B * x = lambda * R * R' * x
        //                      (inv(R) * A * inv(R')) * R' * x = lambda * (R' * x)
        //
        // rocblas_eform_abx    A * B * x = lambda * x
        //                      A * R * R' * x = lambda * inv(R') * R' * x
        //                      (R' * A * R) * (R'* x) = lambda * (R' * x )
        //
        // rocblas_eform_bax    B * A * x = lambda * x
        //                      R * R' * A * x = lambda * x
        //                      (R' * A * R) * inv(R) * x = lambda * inv(R) * x
        // ------------------------------------------------------------------

        bool const is_upper = (uplo == rocblas_fill_upper);
        bool const is_lower = (uplo == rocblas_fill_lower);
        {
            bool const is_valid_uplo = (is_upper || is_lower);
            if(!is_valid_uplo)
            {
                return rocblas_status_invalid_value;
            }
        }

        T t_one = 1;

        rocblas_stride const strideAsave = n * (n - 1) / 2;

        // ---------------------------------------------------
        // symmetrize matrix and save strictly triangular part
        // ---------------------------------------------------
        {
            bool const is_restore = false;

            copy_symm_tri(handle, is_lower, n,

                          A, shiftA, lda, strideA,

                          Asave, strideAsave,

                          batch_count, is_restore);
        }

        if(itype == rocblas_eform_ax)
        {
            // -----------------------------------
            // Compute    A <- inv(L) * A * inv(L')
            // or
            // Compute    A <- inv(U') * A * inv(U)
            // -----------------------------------
            {
                // ----------------
                // A <- inv(L) * A
                // or
                // A <- inv(U') * A
                // ----------------
                rocblas_operation trans
                    = (is_lower) ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
                auto const istat = rocblasCall_trsm(handle, rocblas_side_left, uplo, trans,
                                                    rocblas_diagonal_non_unit,

                                                    n, n,

                                                    &t_one,

                                                    B, shiftB, ldb, strideB,

                                                    A, shiftA, lda, strideA,

                                                    batch_count,

                                                    optim_mem, temp1, temp2, temp3, temp4);
                if(istat != rocblas_status_success)
                {
                    return (istat);
                };
            }

            {
                // ----------------
                // A <- A * inv(L')
                // or
                // A <- A * inv(U)
                // ----------------

                rocblas_operation const trans
                    = (is_lower) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;
                auto const istat = rocblasCall_trsm(handle, rocblas_side_right, uplo, trans,
                                                    rocblas_diagonal_non_unit,

                                                    n, n,

                                                    &t_one,

                                                    B, shiftB, ldb, strideB,

                                                    A, shiftA, lda, strideA,

                                                    batch_count,

                                                    optim_mem, temp1, temp2, temp3, temp4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                };
            }
        }
        else
        {
            rocblas_stride const stride_alpha = 0;
            // ----------------------
            // Compute A <- L' * A * L
            // or
            // Compute A <- U * A * U'
            // ----------------------

            {
                // -----------
                // A <- L' * A
                // or
                // A <- U * A
                // -----------
                rocblas_operation const trans
                    = (is_lower) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;

                auto const istat = rocblasCall_trmm(handle, rocblas_side_left, uplo, trans,
                                                    rocblas_diagonal_non_unit,

                                                    n, n, &t_one, stride_alpha,

                                                    B, shiftB, ldb, strideB,

                                                    A, shiftA, lda, strideA,

                                                    batch_count, (T**)temp4);
                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }

            {
                // ----------
                // A <- A * L
                // or
                // A <- A * U'
                // ----------

                rocblas_operation const trans
                    = (is_lower) ? rocblas_operation_none : rocblas_operation_conjugate_transpose;

                auto const istat = rocblasCall_trmm(handle, rocblas_side_right, uplo, trans,
                                                    rocblas_diagonal_non_unit,

                                                    n, n, &t_one, stride_alpha,

                                                    B, shiftB, ldb, strideB,

                                                    A, shiftA, lda, strideA,

                                                    batch_count, (T**)temp4);
                if(istat != rocblas_status_success)
                {
                    return (istat);
                }
            }
        }

        // -------------------------------
        // restore strictly triangular part
        // -------------------------------
        {
            bool const is_restore = true;
            copy_symm_tri(handle, is_lower, n,

                          A, shiftA, lda, strideA,

                          Asave, strideAsave,

                          batch_count,

                          is_restore);
        }

        return (rocblas_status_success);
    }; // end sygs2_hegs2_alt

    auto call_sygs2_hegs2
        = [sygs2_hegs2_alt, nb, optim_mem, scalars, work_x_temp, workArr_temp_arr, store_wcs_invA,
           invA_arr](

              rocblas_handle handle, rocblas_eform const itype, rocblas_fill const uplo,

              I const n,

              auto A, rocblas_stride const shiftA, I const lda, rocblas_stride const strideA,

              auto B, rocblas_stride const shiftB, I const ldb, rocblas_stride const strideB,

              I const batch_count) {
              if(use_sygs2_hegs2_alt)
              {
                  // ------------------------------------------
                  // note use nb to maintain alignment in temp1
                  // ------------------------------------------
                  size_t const len_Asave = size_t(batch_count) * nb * (nb - 1) / 2;
                  T* const Asave = static_cast<T*>(work_x_temp);

                  // ------------------------
                  // scratch storage for TRSM
                  // ------------------------
                  auto const temp1 = Asave + len_Asave;
                  auto const temp2 = workArr_temp_arr;
                  auto const temp3 = store_wcs_invA;
                  auto const temp4 = invA_arr;

                  auto istat = sygs2_hegs2_alt(handle, itype, uplo, n,

                                               A, shiftA, lda, strideA,

                                               B, shiftB, ldb, strideB, batch_count,

                                               Asave, optim_mem, temp1, temp2, temp3, temp4);
                  if(istat != rocblas_status_success)
                  {
                      return (istat);
                  }
              }
              else
              {
                  auto istat = rocsolver_sygs2_hegs2_template<BATCHED, T>(
                      handle, itype, uplo, n,

                      A, shiftA, lda, strideA,

                      B, shiftB, ldb, strideB,

                      batch_count,

                      scalars, work_x_temp, store_wcs_invA, (T**)workArr_temp_arr);
                  if(istat != rocblas_status_success)
                  {
                      return (istat);
                  }
              }
              return (rocblas_status_success);
          }; // end call_sygs2_hegs2

    // if the matrix is too small, use the unblocked variant of the algorithm
    if(n <= nb)
    {
        return call_sygs2_hegs2(handle, itype, uplo, n,

                                A, shiftA, lda, strideA,

                                B, shiftB, ldb, strideB,

                                batch_count);
    }

    if(itype == rocblas_eform_ax)
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute inv(U')*A*inv(U)
            for(I k = 0; k < n; k += nb)
            {
                I kb = std::min(n - k, nb);
                {
                    auto const istat = call_sygs2_hegs2(handle, itype, uplo, kb,

                                                        A, shiftA + idx2D(k, k, lda), lda, strideA,

                                                        B, shiftB + idx2D(k, k, ldb), ldb, strideB,

                                                        batch_count);
                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }

                if(k + kb < n)
                {
                    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, kb, n - k - kb, B, shiftB + idx2D(k, k, ldb),
                        ldb, strideB, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count,
                        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);

                    rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_syr2k_her2k<BATCHED, T>(
                        handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
                        &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
                        shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);

                    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                        handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                        kb, n - k - kb, B, shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
                }
            }
        }
        else
        {
            // Compute inv(L)*A*inv(L')
            for(I k = 0; k < n; k += nb)
            {
                I kb = std::min(n - k, nb);

                {
                    auto istat = call_sygs2_hegs2(handle, itype, uplo, kb,

                                                  A, shiftA + idx2D(k, k, lda), lda, strideA,

                                                  B, shiftB + idx2D(k, k, ldb), ldb, strideB,

                                                  batch_count);
                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    };
                }

                if(k + kb < n)
                {
                    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                        handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, n - k - kb, kb, B, shiftB + idx2D(k, k, ldb),
                        ldb, strideB, A, shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count,
                        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);

                    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count);

                    rocblasCall_syr2k_her2k<BATCHED, T>(
                        handle, uplo, rocblas_operation_none, n - k - kb, kb, &t_minone, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, B,
                        shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count);

                    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit,
                        n - k - kb, kb, B, shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
                }
            }
        }
    }
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute U*A*U'
            for(I k = 0; k < n; k += nb)
            {
                I kb = std::min(n - k, nb);

                rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none,
                                 rocblas_diagonal_non_unit, k, kb, &t_one, 0, B, shiftB, ldb,
                                 strideB, A, shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, k, kb, &t_one, A,
                    shiftA + idx2D(0, k, lda), lda, strideA, B, shiftB + idx2D(0, k, ldb), ldb,
                    strideB, &s_one, A, shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_trmm(handle, rocblas_side_right, uplo,
                                 rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,
                                 k, kb, &t_one, 0, B, shiftB + idx2D(k, k, ldb), ldb, strideB, A,
                                 shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                {
                    auto istat = call_sygs2_hegs2(handle, itype, uplo, kb, A,
                                                  shiftA + idx2D(k, k, lda), lda, strideA,

                                                  B, shiftB + idx2D(k, k, ldb), ldb, strideB,

                                                  batch_count);
                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }
            }
        }
        else
        {
            // Compute L'*A*L
            for(I k = 0; k < n; k += nb)
            {
                I kb = std::min(n - k, nb);

                rocblasCall_trmm(handle, rocblas_side_right, uplo, rocblas_operation_none,
                                 rocblas_diagonal_non_unit, kb, k, &t_one, 0, B, shiftB, ldb,
                                 strideB, A, shiftA + idx2D(k, 0, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, k, kb, &t_one, A,
                    shiftA + idx2D(k, 0, lda), lda, strideA, B, shiftB + idx2D(k, 0, ldb), ldb,
                    strideB, &s_one, A, shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_trmm(handle, rocblas_side_left, uplo,
                                 rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,
                                 kb, k, &t_one, 0, B, shiftB + idx2D(k, k, ldb), ldb, strideB, A,
                                 shiftA + idx2D(k, 0, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                {
                    auto istat = call_sygs2_hegs2(handle, itype, uplo, kb,

                                                  A, shiftA + idx2D(k, k, lda), lda, strideA,

                                                  B, shiftB + idx2D(k, k, ldb), ldb, strideB,

                                                  batch_count);
                    if(istat != rocblas_status_success)
                    {
                        return (istat);
                    }
                }
            }
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

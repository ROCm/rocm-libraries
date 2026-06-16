/************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack/roclapack_geqrf.hpp"
#include "laset.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
template <bool BATCHED, typename T, typename I>
void rocsolver_sy2sb_he2hb_getMemorySize(const I n,
                                         const I kd,
                                         const I nb,
                                         const I batch_count,
                                         size_t* size_scalars,
                                         size_t* size_D,
                                         size_t* size_V,
                                         size_t* size_W,
                                         size_t* size_X,
                                         size_t* size_Z,
                                         size_t* size_work,
                                         size_t* size_workArr)
{
    *size_scalars = 0;
    *size_D = 0;
    *size_V = 0;
    *size_W = 0;
    *size_X = 0;
    *size_Z = 0;
    *size_work = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0 || kd >= n - 1)
        return;

    size_t w, wa, s1, s2;

    // size for main arrays
    *size_D = sizeof(T) * nb * nb * batch_count;
    *size_V = sizeof(T) * n * nb * batch_count;
    *size_W = sizeof(T) * n * nb * batch_count;
    *size_X = sizeof(T) * n * nb * batch_count;
    *size_Z = sizeof(T) * n * nb * batch_count;
    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : 0;

    // extra space for geqrf calls
    rocsolver_geqrf_getMemorySize<BATCHED, T>(n - kd, kd, batch_count, size_scalars, &w, &s1, &s2,
                                              &wa);
    *size_D = std::max(*size_D, s1);
    *size_Z = std::max(*size_Z, s2);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);

    // extra space for larft calls
    rocsolver_larft_getMemorySize<BATCHED, T>(n - kd, nb, batch_count, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename U>
rocblas_status rocsolver_sy2sb_he2hb_argCheck(rocblas_handle handle,
                                              const I n,
                                              const I kd,
                                              const I nb,
                                              U A,
                                              const I lda,
                                              T* Aband,
                                              const I ldab,
                                              T* tau,
                                              const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || kd < 1 || nb < kd || nb % kd != 0 || lda < n || ldab < 3 * kd - 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !Aband) || (n && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
// Reduces matrix A to Aband with bandwidth kd (inner block size), using outer
// block size nb. We require kd to evenly divide nb. For comparison, LAPACK uses
// single-level blocking with nb == kd.
// Matrix A must be explicitly symmetrized before calling this.
// Householder vectors overwrite A below bandwidth, with associated tau.
// D, V, W, X, Z are workspaces.
template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_sy2sb_he2hb_template(rocblas_handle handle,
                                              const I n,
                                              const I kd,
                                              const I nb,
                                              U A,
                                              const I shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              T* Aband,
                                              const I ldab,
                                              const rocblas_stride strideAb,
                                              T* tau,
                                              const rocblas_stride strideTau,
                                              const I batch_count,
                                              T* scalars,
                                              T* D,
                                              T* V,
                                              T* W,
                                              T* X,
                                              T* Z,
                                              T* work,
                                              T** workArr)
{
    ROCSOLVER_ENTER("sy2sb_he2hb", "n:", n, "kd", kd, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "ldab:", ldab, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // gemm implementation is faster than her2k/hemm,
    // which are provided for comparison.
    bool const use_her2k = false;

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Row of Aband that stores main diagonal.
    I idiag = kd - 1;

    // If band covers matrix, just copy to Aband.
    if(kd >= n - 1)
    {
        // Using ldab-1 converts dense to band format.
        // When kd == n-1, A(0, n-1) is outside the kd-1 upper diagonals,
        // but gets copied to a "don't care" entry in the band structure.
        I cpy_mblks = ceildiv(n, BS2);
        I cpy_nblks = ceildiv(n, BS2);
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_nblks, batch_count),
                                dim3(BS2, BS2), 0, stream, n, n, // opts
                                A, shiftA, lda, strideA, // A
                                Aband, idiag, ldab - 1, strideAb); // Aband
        return rocblas_status_success;
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    T const one = 1;
    T const zero = 0;
    T const neg_half = -0.5;
    T const neg_one = -1;
    S const r_one = 1;

    laset(handle, 'g', ldab, n, zero, zero, Aband, idx2D(0, 0, ldab), ldab, strideAb, batch_count);

    I ldd = nb;
    I ldv = n;
    I ldw = n;
    I ldx = n;
    I ldz = n;

    rocblas_stride strideD = ldd * nb;
    rocblas_stride strideV = ldv * nb;
    rocblas_stride strideW = ldw * nb;
    rocblas_stride strideX = ldx * nb;
    rocblas_stride strideZ = ldz * nb;

    // Index i tracks what sub-panels have been factored.
    I i = 0;
    I cpy_mblks, cpy_nblks;

    // Loop over large blocks.
    for(I j = 0; j < n - kd; j += nb)
    {
        I jm = n - kd - j; // height of outer panel
        I jb = std::min(nb, jm); // width  of outer panel
        I jend = j + jb;

        // Copy panel to factor, to preserve A for hemm.
        // For copying purposes, round up to full kd.
        // Includes diagonal tile above panel and all kd columns.
        I jb_rnd = roundup(jb, kd);
        cpy_mblks = ceildiv(n - j, BS2);
        cpy_nblks = ceildiv(jb_rnd, BS2);
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_nblks, batch_count),
                                dim3(BS2, BS2), 0, stream, n - j, jb_rnd, // opts
                                A, idx2D(j, j, lda) + shiftA, lda, strideA, // Aj
                                V, idx2D(j, 0, ldv), ldv, strideV); // Vj

        // Loop over inner blocking sub-panels to reach bandwidth.
        assert(i == j);
        while(i < jend)
        {
            I qm = n - i - kd;
            I qn = std::min(kd, qm);

            if(i > j)
            {
                // Apply update from previous subpanels.
                // Includes diag tile above panel and all kd columns.
                // Ai -= Vj Zj^H (where Ai in V)
                rocsolver_gemm(handle, rocblas_operation_none,
                               rocblas_operation_conjugate_transpose, n - i, kd, i - j, // opts
                               &neg_one, V, idx2D(i, 0, ldv), ldv, strideV, // Vj
                               Z, idx2D(i, 0, ldz), ldz, strideZ, // Zj^H, kd cols
                               &one, V, idx2D(i, i - j, ldv), ldv, strideV, // Vi
                               batch_count, workArr);

                // Ai -= Zj Vj^H
                rocsolver_gemm(handle, rocblas_operation_none,
                               rocblas_operation_conjugate_transpose, n - i, kd, i - j, // opts
                               &neg_one, Z, idx2D(i, 0, ldz), ldz, strideZ, // Zj
                               V, idx2D(i, 0, ldv), ldv, strideV, // Vj^H, kd cols
                               &one, V, idx2D(i, i - j, ldv), ldv, strideV, // Vi
                               batch_count, workArr);
            }

            // Factor current sub-panel, Ai, stored in Vi.
            // Includes all kd cols, not just qn cols; geqrf updates all cols.
            rocsolver_geqrf_template<BATCHED, STRIDED>(handle, qm, kd, // opts
                                                       V, idx2D(i + kd, i - j, ldv), ldv, strideV, // Vi
                                                       &tau[i], strideTau, // tau_i
                                                       batch_count, scalars, work, D, Z, workArr);

            // Copy band of A (diag tile and R) to Aband.
            // Copies some "don't care" entries from below bandwidth kd.
            // Using ldab-1 converts dense to band format.
            cpy_mblks = ceildiv(kd + 1 + qn, BS2);
            cpy_nblks = ceildiv(kd, BS2);
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_nblks, batch_count),
                                    dim3(BS2, BS2), 0, stream, kd + 1 + qn, kd, // opts
                                    V, idx2D(i, i - j, ldv), ldv, strideV, // A_ii
                                    Aband, idx2D(idiag, i, ldab), ldab - 1, strideAb); // Aband_ii

            // Set upper triangle of Vi to identity.
            T const offdiag = zero;
            T const diag = one;
            laset(handle, 'u', qn, qn, offdiag, diag, // opts
                  V, idx2D(i + kd, i - j, ldv), ldv, strideV, // Vi
                  batch_count);

            // Form corresponding matrix Ti = larft( Vi, tau_i ), stored above Vi.
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise, qm,
                                        qn, // opts
                                        V, idx2D(i + kd, i - j, ldv), ldv, strideV, // Vi
                                        &tau[i], strideTau, // tau_i
                                        V + idx2D(i, i - j, ldv), ldv, strideV, // Ti
                                        batch_count, scalars, work, workArr);

            // Compute Wi = Vi Ti
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, qm, qn, qn, // opts
                           &one, V, idx2D(i + kd, i - j, ldv), ldv, strideV, // Vi
                           V, idx2D(i, i - j, ldv), ldv, strideV, // Ti
                           &zero, W, idx2D(i + kd, i - j, ldw), ldw, strideW, // Wi
                           batch_count, workArr);

            if(i > j)
            {
                // Update Wi with contributions from previous sub-panels.
                // Wi = Wi* - Wj Cji, where Cji = Vj^H Wi*
                // and Wi* = Vi Ti is current value of Wi.
                // Note Tji = -Tj Cji, if we want to later compute entire
                // T = [ Tj  Tji ].
                //     [ 0   Ti  ]

                // Zero out block above Wi*.
                laset(handle, 'g', i - j, qn, zero, zero, // opts
                      W, idx2D(j + kd, i - j, ldw), ldw, strideW, // Wi
                      batch_count);

                // Cji = Vj^H Wi, Cji stored in V above [ Ti; Vi ].
                rocsolver_gemm(handle, rocblas_operation_conjugate_transpose,
                               rocblas_operation_none, i - j, qn, qm, // opts
                               &one, V, idx2D(i + kd, 0, ldv), ldv, strideV, // Vj^H
                               W, idx2D(i + kd, i - j, ldw), ldw, strideW, // Wi
                               &zero, V, idx2D(j, i - j, ldv), ldv, strideV, // Cji
                               batch_count, workArr);

                // Wi = Wi* - Wj Cji
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, jm, qn,
                               i - j, // opts
                               &neg_one, W, idx2D(j + kd, 0, ldw), ldw, strideW, // Wj
                               V, idx2D(j, i - j, ldv), ldv, strideV, // Cji
                               &one, W, idx2D(j + kd, i - j, ldw), ldw, strideW, // Wi, jm rows
                               batch_count, workArr);
            }

            // Prepare Hermitian rank-2k update.
            // Xi = A Wi
            // Because Wi is coupled with Wj, it is jm rows tall instead of qm.
            if constexpr(use_her2k)
            {
                rocblasCall_symm_hemm(handle, rocblas_side_left, rocblas_fill_lower, jm, qn, // opts
                                      &one, A, idx2D(j + kd, j + kd, lda) + shiftA, lda, strideA, // A
                                      W, idx2D(j + kd, i - j, ldw), ldw, strideW, // Wi, jm rows
                                      &zero, X, idx2D(j + kd, i - j, ldx), ldx, strideX, // Xi
                                      batch_count);
            }
            else
            {
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, jm, qn,
                               jm, // opts
                               &one, A, idx2D(j + kd, j + kd, lda) + shiftA, lda, strideA, // A
                               W, idx2D(j + kd, i - j, ldw), ldw, strideW, // Wi, jm rows
                               &zero, X, idx2D(j + kd, i - j, ldx), ldx, strideX, // Xi
                               batch_count, workArr);
            }

            // D = Wj^H Xj = Wj^H (A Wj)
            // D is Hermitian, so this could be herkx/gemmtr, with a hemm below for Z.
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
                           i - j + qn, i - j + qn, jm, // opts
                           &one, W, idx2D(j + kd, 0, ldw), ldw, strideW, // Wj^H
                           X, idx2D(j + kd, 0, ldx), ldx, strideX, // Xj
                           &zero, D, idx2D(0, 0, ldd), ldd, strideD, // D
                           batch_count, workArr);

            // Zj = Xj - 0.5 Vj D
            //    = Xj - 0.5 Vj Wj^H A Wj
            //    = A V T - 0.5 V T^H V^H A V T
            // Zj is really jm rows tall, but we need only qm rows for her2k/gemms
            // to update the next panel (above) or trailing matrix (below).
            // Too bad there isn't a 4 matrix gemm: C = alpha AB + beta D.
            cpy_mblks = ceildiv(qm, BS2);
            cpy_nblks = ceildiv(i - j + qn, BS2);
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_nblks, batch_count),
                                    dim3(BS2, BS2), 0, stream, qm, i - j + qn, // opts
                                    X, idx2D(i + kd, 0, ldx), ldx, strideX, // Xj
                                    Z, idx2D(i + kd, 0, ldz), ldz, strideZ); // Zj
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, qm, i - j + qn,
                           i - j + qn, // opts
                           &neg_half, V, idx2D(i + kd, 0, ldv), ldv, strideV, // Vj
                           D, idx2D(0, 0, ldd), ldd, strideD, // D
                           &one, Z, idx2D(i + kd, 0, ldz), ldz, strideZ, // Zj
                           batch_count, workArr);

            i += kd;
        }

        // Update trailing matrix.
        //      A := Q^H A Q
        //         = (I - VTV^H)^H A (I - VTV^H)
        //         = A - AVT V^H - V T^H V^H A + V T^H V^H AVT V^H
        //         = A - (AVT) V^H - V (AVT)^H + 0.5 V (T^H V^H AVT V^H) + 0.5 (V T^H V^H AVT) V^H *
        //         = A - (AVT - 0.5 V T^H V^H AVT) V^H - V (AVT - 0.5 V T^H V^H AVT)^H *
        //         = A - ZV^H - VZ^H    her2k or (2x) gemm
        // where Z = AVT - 0.5 V T^H V^H AVT, computed above as:
        //       W = V T                trmm or gemm (gemm requires T to have explicit 0's)
        //       X = A W = A V T        hemm or gemm (gemm requires A to be symmetrized)
        //       D = W^H X = W^H A W    herkx/gemmtr or gemm
        //       Z = X - 0.5 V D        hemm or gemm
        // Several gemms require V to have explicit 1's and 0's.
        // * These steps apply fact that A is Hermitian.
        assert(i < n);
        if constexpr(use_her2k)
        {
            // A -= ZV^H + VZ^H
            rocblasCall_syr2k_her2k<BATCHED, T>(
                handle, rocblas_fill_lower, rocblas_operation_none, n - i, jb, // opts
                &neg_one, Z, idx2D(i, 0, ldz), ldz, strideZ, // Zj, n-i rows
                V, idx2D(i, 0, ldv), ldv, strideV, // Vj, n-i rows
                &r_one, A, idx2D(i, i, lda), lda, strideA, // Ai
                batch_count);
        }
        else
        {
            // A -= VZ^H
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - i, n - i, jb, // opts
                           &neg_one, V, idx2D(i, 0, ldv), ldv, strideV, // Vj,   n-i rows
                           Z, idx2D(i, 0, ldz), ldz, strideZ, // Zj^H, n-i cols
                           &one, A, idx2D(i, i, lda) + shiftA, lda, strideA, // Ai
                           batch_count, workArr);

            // A -= ZV^H
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - i, n - i, jb, // opts
                           &neg_one, Z, idx2D(i, 0, ldz), ldz, strideZ, // Zj,   n-i rows
                           V, idx2D(i, 0, ldv), ldv, strideV, // Vj^H, n-i cols
                           &one, A, idx2D(i, i, lda) + shiftA, lda, strideA, // Ai
                           batch_count, workArr);
        }

        // Copy factored panel with all [Cij; Ti; Vi] back to A.
        // If we don't need [Cij, Ti], this could be reduced to n-j-kd rows.
        // This could be done in parallel with above trailing matrix update.
        cpy_mblks = ceildiv(n - j, BS2);
        cpy_nblks = ceildiv(jb_rnd, BS2);
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_nblks, batch_count),
                                dim3(BS2, BS2), 0, stream, n - j, jb_rnd, // opts
                                V, idx2D(j, 0, ldv), ldv, strideV, // Vj
                                A, idx2D(j, j, lda) + shiftA, lda, strideA); // Aj
    }

    // Copy last, lower triangular block of band of A to Aband.
    cpy_mblks = ceildiv(n - i, BS2);
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(cpy_mblks, cpy_mblks, batch_count), dim3(BS2, BS2), 0,
                            stream, n - i, n - i, // opts
                            A, idx2D(i, i, lda) + shiftA, lda, strideA, // Aii
                            Aband, idx2D(idiag, i, ldab), ldab - 1, strideAb, // Aband_ii
                            no_mask{}, rocblas_fill_lower);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

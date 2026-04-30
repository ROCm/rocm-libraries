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
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "laset.hpp"

#include "print_matrix.hpp"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
template <bool BATCHED, typename T>
void rocsolver_sy2sb_he2hb_getMemorySize(
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int nb,
    const rocblas_int batch_count,
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
    if(n == 0 || batch_count == 0 || kd == 0 || nb == 0)
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
    /// todo: this was ignoring `w`?
    rocsolver_geqrf_getMemorySize<BATCHED, T>(n - kd, kd, batch_count, size_scalars, &w, &s1, &s2, &wa);
    *size_D = std::max(*size_D, s1);
    *size_Z = std::max(*size_Z, s2);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);

    // extra space for larft calls
    /// todo: was n-nb, but seems it should be n-kd.
    rocsolver_larft_getMemorySize<BATCHED, T>(n - kd, nb, batch_count, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);
}

//------------------------------------------------------------------------------
/// Shouldn't these be T* A, tau instead of T A, tau?
// todo: this was template <typename T, typename S> with T A, S V, S W.
// What's the difference between T and S? For complex, they are all complex, not real.
// Why no pointers?
template <typename T>
rocblas_status rocsolver_sy2sb_he2hb_argCheck(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int nb,
    T* A,
    const rocblas_int lda,
    T* Aband,
    const rocblas_int ldab,
    T* tau,
    const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || (n > 0 && kd < 1) || nb < kd || nb % kd != 0 || lda < n || batch_count < 0)
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
/// What's the point of types T and U? Isn't T == U always?
/// Reduces A to Aband with bandwidth kd, using outer block size nb.
/// Householder vectors overwrite A below bandwidth, with associated tau.
/// T, U, V, W, Z are workspaces.
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_sy2sb_he2hb_template(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int nb,
    U A,
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    T* Aband,
    const rocblas_int ldab,
    const rocblas_stride strideAb,
    T* tau,
    const rocblas_stride strideTau,
    const rocblas_int batch_count,
    T* scalars,
    T* D, const rocblas_int ldd,
    T* V, const rocblas_int ldv,
    T* W, const rocblas_int ldw,
    T* X, const rocblas_int ldx,
    T* Z, const rocblas_int ldz,
    T* work,
    T** workArr)
{
    ROCSOLVER_ENTER("sy2sb_he2hb", "n:", n, "kd", kd, "nb:", nb,
                    "shiftA:", shiftA, "lda:", lda, "ldab:", ldab,
                    "bc:", batch_count);

    const bool debug_ = false;
    if (debug_)
        printf( "he2hb n %d, kd %d, nb %d, lda %d, ldab %d\n", n, kd, nb, lda, ldab );

    using S = decltype(std::real(T{}));

    bool const use_her2k = false;

    // quick return
    if(n == 0 || kd == 0 || nb == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T const one = 1;
    T const zero = 0;
    T const neghalf = -0.5;
    T const negone = -1;
    S const rone = 1;

    if (debug_)
        print_matrix( "A_in", n, n, A, lda );
    laset(
        handle, 'g',
        ldab, n, zero, zero,
        Aband, idx2D( 0, 0, ldab ), ldab, strideAb,
        batch_count );
    if (debug_)
        print_matrix( "Aband_in", ldab, n, Aband, ldab );

    rocblas_stride strideD = ldd*nb;
    rocblas_stride strideV = ldv*nb;
    rocblas_stride strideW = ldw*nb;
    rocblas_stride strideX = ldx*nb;
    rocblas_stride strideZ = ldz*nb;

    // Row of Aband that stores main diagonal.
    rocblas_int idiag = kd - 1;

    // Index i tracks what sub-panels have been factored.
    rocblas_int i = 0;
    rocblas_int cpy_mblks, cpy_nblks;

    // Loop over large blocks.
    for (rocblas_int j = 0; j < n - kd; j += nb)
    {
        rocblas_int jm = n - kd - j;          // height of outer panel
        rocblas_int jb = std::min( nb, jm );  // width  of outer panel
        rocblas_int jend = j + jb;

        if (debug_)
            printf( "----------\nj = %d, jb = %d, jend = %d, jm = %d\n", j, jb, jend, jm );

        // Copy panel to factor, to preserve A for hemm.
        // For copying purposes, round up to full kd.
        // Includes diagonal tile above panel and all kd columns.
        // (minor todo: could copy 1st diagonal tile (j:j+kb) to Aband instead,
        // but then later code to copy panel to Aband is more complex.)
        rocblas_int jb_rnd = roundup( jb, kd );
        cpy_mblks = ceildiv( n-j, 32 );
        cpy_nblks = ceildiv( jb_rnd, 32 );
        ROCSOLVER_LAUNCH_KERNEL(
            copy_mat<T>,
            dim3(cpy_mblks, cpy_nblks, batch_count), dim3(32, 32), 0, stream,
            n-j, jb_rnd,
            A, idx2D( j, j, lda ) + shiftA, lda, strideA,  // Aj
            V, idx2D( j, 0, ldv ), ldv, strideA );         // Vj

        if (debug_)
            print_matrix( "Vj", n, nb, V, ldv, 3, stream );

        // Loop over inner blocking sub-panels to reach bandwidth.
        assert( i == j );
        while (i < jend)
        {
            rocblas_int qm = n - i - kd;
            rocblas_int qn = std::min( kd, qm );

            if (debug_)
                printf( "-----\ni = %d, qm = %d, qn = %d\n", i, qm, qn );

            if (i > j)
            {
                // Apply update from previous subpanels.
                // Includes diag tile above panel and all kd columns.
                // Ai -= Vj Zj^H (where Ai in V)
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                    n-i, kd, i-j,
                    &negone, V, idx2D( i, 0,   ldv ), ldv, strideV,  // Vj
                             Z, idx2D( i, 0,   ldz ), ldz, strideZ,  // Zj^H, kd cols
                    &one,    V, idx2D( i, i-j, ldv ), ldv, strideV,  // Vi
                    batch_count, workArr );

                // Ai -= Zj Vj^H
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                    n-i, kd, i-j,
                    &negone, Z, idx2D( i, 0,   ldz ), ldz, strideZ,  // Zj
                             V, idx2D( i, 0,   ldv ), ldv, strideV,  // Vj^H, kd cols
                    &one,    V, idx2D( i, i-j, ldv ), ldv, strideV,  // Vi
                    batch_count, workArr );
            }

            // Factor current sub-panel, Ai, stored in Vi.
            // Includes all kd cols, not just qn cols; geqrf updates all cols.
            rocsolver_geqrf_template<BATCHED, STRIDED>(
                handle, qm, kd,
                V, idx2D( i+kd, i-j, ldv ), ldv, strideV,  // Vi
                &tau[ i ], strideTau,                      // tau_i
                batch_count, scalars, work, D, Z, workArr );

            if (debug_) {
                print_matrix( "Vi (R)", n, nb, V, ldv, 3, stream );
                //print_matrix( "Vi (R)", n, kd, V + idx2D( 0, i-j, ldv ), ldv, 3, stream );
            }

            // Copy band of A (diag tile and R) to Aband.
            // Copies some "don't care" entries from below bandwidth kd.
            // Using ldab-1 converts dense to band format.
            cpy_mblks = ceildiv( kd+1+qn, 32 );
            cpy_nblks = ceildiv( kd, 32 );
            ROCSOLVER_LAUNCH_KERNEL(
                copy_mat<T>,
                dim3(cpy_mblks, cpy_nblks, batch_count), dim3(32, 32), 0, stream,
                kd+1+qn, kd,
                V, idx2D( i, i-j, ldv ), ldv, strideV,
                Aband, idx2D( idiag, i, ldab ), ldab-1, strideAb );

            if (debug_)
                print_matrix( "Aband_i", ldab, n, Aband, ldab, 3, stream );

            // Set upper triangle of Vi to identity.
            T const offdiag = zero;
            T const diag    = one;
            laset(
                handle, 'u',
                qn, qn, offdiag, diag,
                V, idx2D( i+kd, i-j, ldv ), ldv, strideV,  // Vi
                batch_count );

            if (debug_) {
                print_matrix( "Vi (I)", n, nb, V, ldv, 3, stream );
                //print_matrix( "Vi (I)", n, kd, V + idx2D( 0, i-j, ldv ), ldv, 3, stream );
            }

            // Form corresponding matrix Ti = larft( Vi, tau_i ), stored above Vi.
            // todo: why does T not have shift? Is just adding okay? Why not add everywher?
            rocsolver_larft_template<T>(
                handle, rocblas_forward_direction, rocblas_column_wise,
                qm, qn,
                V, idx2D( i+kd, i-j, ldv ), ldv, strideV,  // Vi
                &tau[ i ], strideTau,                      // tau_i
                V + idx2D( i, i-j, ldv ), ldv, strideV,    // Ti
                batch_count, scalars, work, workArr );

            // Compute Wi = Vi Ti
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, qn, qn,
                &one,  V, idx2D( i+kd, i-j, ldv ), ldv, strideV,  // Vi
                       V, idx2D( i,    i-j, ldv ), ldv, strideV,  // Ti
                &zero, W, idx2D( i+kd, i-j, ldw ), ldw, strideW,  // Wi
                batch_count, workArr );

            if (i > j)
            {
                // Update Wi with contributions from previous sub-panels.
                // Wi = Wi* - Wj Cji, where Cji = Vj^H Wi*
                // and Wi* = Vi Ti is current value of Wi.
                // Note Tji = -Tj Cji, if we want to later compute entire
                // T = [ Tj  Tji ].
                //     [ 0   Ti  ]

                // Zero out block above Wi*.
                laset(
                    handle, 'g',
                    i-j, qn, zero, zero,
                    W, idx2D( j+kd, i-j, ldw ), ldw, strideW,  // Wi
                    batch_count );

                // Cji = Vj^H Wi, Cji stored in V above [ Ti; Vi ].
                rocsolver_gemm(
                    handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
                    i-j, qn, qm,
                    &one,  V, idx2D( i+kd, 0,   ldv ), ldv, strideV,  // Vj^H
                           W, idx2D( i+kd, i-j, ldw ), ldw, strideW,  // Wi
                    &zero, V, idx2D( j,    i-j, ldv ), ldv, strideV,  // Cji
                    batch_count, workArr );

                // Wi = Wi* - Wj Cji
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_none,
                    jm, qn, i-j,
                    &negone, W, idx2D( j+kd, 0,   ldw ), ldw, strideW,  // Wj
                             V, idx2D( j,    i-j, ldv ), ldv, strideV,  // Cji
                    &one,    W, idx2D( j+kd, i-j, ldw ), ldw, strideW,  // Wi, jm rows
                    batch_count, workArr );
            }

            // Prepare Hermitian rank-2k update.
            // Xi = A Wi
            // Because Wi is coupled with Wj, it is jm rows tall instead of qm.
            if constexpr (use_her2k) {
                rocsolver_hemm(
                    handle, rocblas_side_left, rocblas_fill_lower,
                    jm, qn,
                    &one,  A, idx2D( j+kd, j+kd, lda ) + shiftA, lda, strideA,  // A
                           W, idx2D( j+kd, i-j,  ldw ), ldw, strideW,  // Wi, jm rows
                    &zero, X, idx2D( j+kd, i-j,  ldx ), ldx, strideX,  // Xi
                    batch_count, workArr );
            }
            else {
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_none,
                    jm, qn, jm,
                    &one,  A, idx2D( j+kd, j+kd, lda ) + shiftA, lda, strideA,  // A
                           W, idx2D( j+kd, i-j,  ldw ), ldw, strideW,  // Wi, jm rows
                    &zero, X, idx2D( j+kd, i-j,  ldx ), ldx, strideX,  // Xi
                    batch_count, workArr );
            }

            // D = Wj^H Xj = Wj^H (A Wj)
            // D is Hermitian, so this could be herkx/gemmtr, with a hemm below for Z.
            rocsolver_gemm(
                handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
                i-j+qn, i-j+qn, jm,
                &one,  W, idx2D( j+kd, 0, ldw ), ldw, strideW,  // Wj^H
                       X, idx2D( j+kd, 0, ldx ), ldx, strideX,  // Xj
                &zero, D, idx2D( 0,    0, ldd ), ldd, strideD,  // D
                batch_count, workArr );

            // Zj = Xj - 0.5 Vj D
            //    = Xj - 0.5 Vj Wj^H A Wj
            //    = A V T - 0.5 V T^H V^H A V T
            // Zj is really jm rows tall, but we need only qm rows for her2k/gemms to update the next panel (above) or trailing matrix (below).
            // Too bad there isn't a 4 matrix gemm: C = alpha AB + beta D.
            cpy_mblks = ceildiv( qm, 32 );
            cpy_nblks = ceildiv( i-j+qn, 32 );
            ROCSOLVER_LAUNCH_KERNEL(
                copy_mat<T>,
                dim3(cpy_mblks, cpy_nblks, batch_count), dim3(32, 32), 0, stream,
                qm, i-j+qn,
                X, idx2D( i+kd, 0, ldx ), ldx, strideX,    // Xj
                Z, idx2D( i+kd, 0, ldz ), ldz, strideZ );  // Zj
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, i-j+qn, i-j+qn,
                &neghalf, V, idx2D( i+kd, 0, ldv ), ldv, strideV,  // Vj
                          D, idx2D( 0,    0, ldd ), ldd, strideD,  // D
                &one,     Z, idx2D( i+kd, 0, ldz ), ldz, strideZ,  // Zj
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
        assert( i < n );
        if constexpr (use_her2k) {
            // A -= ZV^H + VZ^H
            rocsolver_her2k(
                handle, rocblas_fill_lower, rocblas_operation_none,
                n-i, jb,
                &negone, Z, idx2D( i, 0, ldz ), ldz,  // Zj, n-i rows
                         V, idx2D( i, 0, ldv ), ldv,  // Vj, n-i rows
                &one,    A, idx2D( i, i, lda ), lda,  // Ai
                batch_count, workArr);
        }
        else {
            // A -= VZ^H
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                n-i, n-i, jb,
                &negone, V, idx2D( i, 0, ldv ), ldv, strideV,  // Vj,   n-i rows
                         Z, idx2D( i, 0, ldz ), ldz, strideZ,  // Zj^H, n-i cols
                &one,    A, idx2D( i, i, lda ) + shiftA, lda, strideA,  // Ai
                batch_count, workArr);

            // A -= ZV^H
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                n-i, n-i, jb,
                &negone, Z, idx2D( i, 0, ldz ), ldz, strideZ,  // Zj,   n-i rows
                         V, idx2D( i, 0, ldv ), ldv, strideV,  // Vj^H, n-i cols
                &one,    A, idx2D( i, i, lda ) + shiftA, lda, strideA,  // Ai
                batch_count, workArr);
        }

        // Copy factored panel with all [Cij; Ti; Vi] back to A.
        // If we don't need [Cij, Ti], this could be reduced to n-j-kd rows.
        // This could be done in parallel with above trailing matrix update.
        cpy_mblks = ceildiv( n-j, 32 );
        cpy_nblks = ceildiv( jb_rnd, 32 );
        ROCSOLVER_LAUNCH_KERNEL(
            copy_mat<T>,
            dim3(cpy_mblks, cpy_nblks, batch_count), dim3(32, 32), 0, stream,
            n-j, jb_rnd,
            V, idx2D( j, 0, ldv ), ldv, strideV,    // Vj
            A, idx2D( j, j, lda ), lda, strideA );  // Aj
    }

    // Copy last, lower triangular block of band of A to Aband.
    // Using ldab-1 converts dense to band format.
    cpy_mblks = ceildiv( n-i, 32 );
    ROCSOLVER_LAUNCH_KERNEL(
        copy_mat<T>,
        dim3(cpy_mblks, cpy_mblks, batch_count), dim3(32, 32), 0, stream,
        n-i, n-i,
        A,     idx2D( i, i, lda  ) + shiftA, lda,    strideA,   // Aii
        Aband, idx2D( idiag, i, ldab ),      ldab-1, strideAb,  // Aband_ii
        no_mask{}, rocblas_fill_lower );

    if (debug_) {
        print_matrix( "A", n, n, A, lda );
        print_matrix( "Aband", ldab, n, Aband, ldab );
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

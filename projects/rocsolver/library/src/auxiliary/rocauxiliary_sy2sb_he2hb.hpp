/************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

 #include "lapack/roclapack_gelqf.hpp"
 #include "lapack/roclapack_geqrf.hpp"
 #include "rocblas.hpp"
 #include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
// Sets V[ 0:inc, : ] = 0. (Python : rules.)
//      V[ strictly-upper ] = 0.
//      V[ diag           ] = 1.
//      A[ lower ] = A[ upper ]^T (no conj)
//      A[ ...   ] = 0
template <typename T, typename U>
ROCSOLVER_KERNEL void sy2sb_updateAV_kernel(
    const rocblas_int inc,
    const rocblas_int kd,
    const rocblas_int m,
    const rocblas_int n,
    U A,
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    T* V,
    const rocblas_int ldv,
    const rocblas_stride strideV)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Vp = load_ptr_batch<T>(V, b, 0, strideV);

    /// These don't make any sense to me. Won't this access A out-of-bounds?
    /// Also, why not just leave V inside A?
    /// Bcs need to set diagonal of V to 0's and 1's for gemm.

    rocblas_int iv = i;
    rocblas_int jv = j + inc;
    rocblas_int ia = i + kd;
    rocblas_int ja = j + inc;
    if(i < m && j < n)
    {
        if(i < inc)
        {
            Vp[iv + jv * ldv] = 0;
        }
        else
        {
            rocblas_int ib = i - inc;
            T val = Ap[ia + ja * lda];
            if(ib < j)
            {
                Ap[ja + ia * lda] = val;
                Vp[iv + jv * ldv] = 0;
            }
            else if(ib == j)
            {
                Ap[ja + ia * lda] = val;
                Vp[iv + jv * ldv] = 1;
            }
            else
            {
                Ap[ia + ja * lda] = 0;
                Ap[ja + ia * lda] = 0;
                Vp[iv + jv * ldv] = val;
            }
        }
    }
}

//------------------------------------------------------------------------------
template <bool BATCHED, typename T>
void rocsolver_sy2sb_he2hb_getMemorySize(
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int nb,
    const rocblas_int batch_count,
    size_t* size_scalars,
    size_t* size_Acpy,
    size_t* size_work,
    size_t* size_workT,
    size_t* size_workZ,
    size_t* size_workArr)
{
    *size_scalars = 0;
    *size_Acpy = 0;
    *size_work = 0;
    *size_workT = 0;
    *size_workZ = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0 || kd == 0 || nb == 0)
        return;

    rocblas_int n_kd = n - kd;
    size_t w, wa, s1, s2;

    // size for main arrays
    *size_Acpy = sizeof(T) * n_kd * (n_kd + 1) * batch_count;
    *size_workT = sizeof(T) * nb * nb * batch_count;
    *size_workZ = sizeof(T) * n_kd * nb * batch_count;
    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : 0;

    // extra space for geqrf calls
    rocsolver_geqrf_getMemorySize<BATCHED, T>(n_kd, kd, batch_count, size_scalars, &w, &s1, &s2, &wa);
    *size_workT = std::max(*size_workT, s1);
    *size_workZ = std::max(*size_workZ, s2);
    *size_workArr = std::max(*size_workArr, wa);

    // extra space for larft calls
    rocsolver_larft_getMemorySize<BATCHED, T>(n-nb, nb, batch_count, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);
}

//------------------------------------------------------------------------------
template <typename T, typename S>
rocblas_status rocsolver_sy2sb_he2hb_argCheck(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int nb,
    T A,
    const rocblas_int lda,
    S V,
    S W,
    const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || (n > 0 && kd < 1) || nb < kd || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !V) || (n && !W))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
/// What's the point of types T and U? Isn't T == U always?
/// Reduces A to Aband with bandwidth kd, using block size nb.
/// Outputs Householder vectors in V.
/// W is workspace.
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
    T* V,
    const rocblas_int ldv,
    const rocblas_stride strideV,
    T* W,
    const rocblas_int ldw,
    const rocblas_stride strideW,
    const rocblas_int batch_count,
    T* scalars,
    T* Acpy,
    T* work,
    T* workT,
    T* workZ,
    T** workArr)
{
    ROCSOLVER_ENTER("sy2sb_he2hb", "n:", n, "kd", kd, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || kd == 0 || nb == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T zero = 0;
    T neghalf = -0.5;
    T negone = -1;
    S rone = 1;

    rocblas_int ldt = nb;
    rocblas_stride strideT = nb * nb;
    rocblas_int n_kd = n - kd;
    rocblas_int ldacpy = n_kd;
    rocblas_stride strideAcpy = n_kd * (n_kd + 1);
    rocblas_stride strideP = strideAcpy;
    rocblas_int ldz = n_kd;
    rocblas_stride strideZ = n_kd * nb;
    T* tau = Acpy + n_kd * n_kd;

    // Loop over large blocks.
    for(rocblas_int i = 0; i < n_kd; i += nb)
    {
        rocblas_int qm = n_kd - i;
        rocblas_int qn = std::min(kd, qm);
        rocblas_int endb = std::min(i+nb, n_kd);
        rocblas_int kk = qn;
        rocblas_int j, inc, qnn, qmm;

        // keep copy of trailing matrix in Acpy to update V and W
        /// Why is this needed? Seems like large overhead to copy trailing matrix.
        /// Which kernel does this call? Doesn't seem to exactly match a copy_mat kernel that I see.
        /// Does this symmetrize A?
        rocblas_int cpy_blks = (qm - 1) / 32 + 1;
        /// Launch: gridDim (ceildiv( qm, 32 ), -, batch), blockDim (32 x 32), shared mem (0), stream
        ROCSOLVER_LAUNCH_KERNEL(
            (copy_mat<T>),
            dim3(cpy_blks, cpy_blks, batch_count), dim3(32, 32), 0, stream,
            qm, qm,
            A, shiftA + idx2D(i + kd, i + kd, lda), lda, strideA,
            Acpy, 0, ldacpy, strideAcpy);

        /// TODO can this be merged into for j loop? Almost surely.

        // reduce first panel in block
        /// TODO: qm x qn, instead of x kd?
        /// Maybe geqrf limits n to min( m, n ), but I don't think it should.
        rocsolver_geqrf_template<BATCHED, STRIDED>(
            handle, qm, kd,
            A, shiftA + idx2D(i + kd, i, lda), lda, strideA,
            tau, strideP, batch_count, scalars, work, workT, workZ, workArr);

        // Form corresponding matrix T
        rocsolver_larft_template<T>(
            handle, rocblas_forward_direction, rocblas_column_wise,
            qm, qn,
            A, shiftA + idx2D(i + kd, i, lda), lda, strideA,
            tau, strideP, workT, ldt, strideT, batch_count, scalars, work, workArr);

        // update A and V
        /// TODO: qm x qn, instead of x kd?
        rocblas_int mblks = (qm - 1) / 32 + 1;
        rocblas_int nblks = (kd - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(
            (sy2sb_updateAV_kernel),
            dim3(mblks, nblks, batch_count), dim3(32, 32), 0, stream,
            0, kd, qm, kd,
            A, shiftA + idx2D(i, i, lda), lda, strideA,
            V + idx2D(i, i, ldv), ldv, strideV);

        /// Apply 2-sided update with Q:
        ///     A := Q^H A Q
        ///        = (I - VTV^H)^H A (I - VTV^H)
        ///        = (I - V T^H V^H) A (I - VTV^H)
        ///        = (A - V T^H V^H A) (I - VTV^H)
        ///        = A - AVT V^H - V T^H V^H A + V T^H V^H AVT V^H
        ///        = A - (AVT) V^H - V (AVT)^H + 0.5 V (T^H V^H AVT V^H) + 0.5 (V T^H V^H AVT) V^H *
        ///        = A - (AVT - 0.5 V T^H V^H AVT) V^H - V (AVT - 0.5 V T^H V^H AVT)^H *
        ///        = A - ZV^H - VZ^H    her2k
        /// where  Z  = AVT - 0.5 V T^H V^H AVT,
        ///        W  = V T                     gemm; LAPACK would do trmm? Like trtzmm.
        ///        Z0 = A W = A V T             gemm (requires A symmetrized); LAPACK would do hemm.
        ///        T2 = W^H Z0 = T^H V^H A V T  gemm; LAPACK would do trmm + gemm?
        ///        Z  = Z0 - 0.5 V T2 = above   gemm; LAPACK would do trmm + gemm?
        /// * These steps rely on A == A^H.
        /// Note T and T2 are unrelated, they just use the same workspace.

        // Update W = V T
        // sizes (qm x qn) = (qm x qn) (qn x qn), qn is this block size (jb); could be trmm right?
        rocsolver_gemm(
            handle, rocblas_operation_none, rocblas_operation_none,
            qm, qn, qn,
            &one,  V, idx2D(i, i, ldv), ldv, strideV,
                   workT, 0, ldt, strideT,
            &zero, W, idx2D(i, i, ldw), ldw, strideW, batch_count, workArr);

        // prepare symmetric rank update
        // Z0 = A W
        // sizes (qm x qn) = (qm x qm) (qm x qn); could be hemm
        /// TODO: This assumes A is symmetrized. Otherwise NEED hemm!
        rocsolver_gemm(
            handle, rocblas_operation_none, rocblas_operation_none,
            qm, qn, qm,
            &one,  Acpy, 0, ldacpy, strideAcpy,
                   W, idx2D(i, i, ldw), ldw, strideW,
            &zero, workZ, 0, ldz, strideZ, batch_count, workArr);
        // T2 = W^H Z0
        // sizes (qn x qn) = (qm x qn)^H (qm x qn); block inner product
        // reusing T workspace.
        rocsolver_gemm(
            handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
            qn, qn, qm,
            &one,  W, idx2D(i, i, ldw), ldw, strideW,
                   workZ, 0, ldz, strideZ,
            &zero, workT, 0, ldt, strideT, batch_count, workArr);
        // Z = Z0 - 0.5 V T2 = Z - 0.5 V (W^H Z) = (A W) - 0.5 V (W^H A W)
        //   = A V T - 0.5 V T^H V^H A V T
        rocsolver_gemm(
            handle, rocblas_operation_none, rocblas_operation_none,
            qm, qn, qn,
            &neghalf, V, idx2D(i, i, ldv), ldv, strideV,
                      workT, 0, ldt, strideT,
            &one,     workZ, 0, ldz, strideZ, batch_count, workArr);

        // Inner blocking to reach bandwidth.
        // reduce all other panels in block
        for (j = i+kd; j < endb; j += kd)
        ///j = i + kd;
        ///while(j < endb)
        {
            inc = j - i;
            qmm = n_kd - j;
            qnn = std::min(kd, qmm);
            kk += qnn;

            /// if (j > i) {
                // update current panel
                // Left looking: apply updates from previous sub-panels.
                // A = A - V Z^H
                /// TODO: qnn instead of kd?
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                    qmm + kd, kd, inc,
                    &negone, V, idx2D(i+inc-kd, i, ldv), ldv, strideV,
                             workZ, inc-kd, ldz, strideZ,
                    &one,    A, shiftA + idx2D(j, j, lda), lda, strideA, batch_count, workArr);
                // A = A - Z V^H
                rocsolver_gemm(
                    handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                    qmm + kd, kd, inc,
                    &negone, workZ, inc-kd, ldz, strideZ,
                             V, idx2D(i+inc-kd, i, ldv), ldv, strideV,
                    &one,    A, shiftA + idx2D(j, j, lda), lda, strideA, batch_count, workArr);
            /// } // end if

            // reduce current panel
            rocsolver_geqrf_template<BATCHED, STRIDED>(
                handle, qmm, kd,
                A, shiftA + idx2D(j + kd, j, lda), lda, strideA,
                tau, strideP, batch_count, scalars, work, workT, workZ, workArr);

            // Form corresponding matrix Tj = larft( Vj, tau_j )
            /// How is the whole tau being saved outside this function?
            /// Don't we need that for the back transform (unmtr / unmqr)?
            rocsolver_larft_template<T>(
                handle, rocblas_forward_direction, rocblas_column_wise,
                qmm, qnn,
                A, shiftA + idx2D(j + kd, j, lda), lda, strideA,
                tau, strideP, workT, ldt, strideT, batch_count, scalars, work, workArr);

            // update A and V
            /// TODO qmm x qnn?
            ROCSOLVER_LAUNCH_KERNEL(
                (sy2sb_updateAV_kernel),
                dim3(mblks, nblks, batch_count), dim3(32, 32), 0, stream,
                inc, kd, qm, kd,
                A, shiftA + idx2D(i, i, lda), lda, strideA,
                V + idx2D(i, i, ldv), ldv, strideV);

            // update W
            // TODO qmm?
            // Wj = Vj Tj
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, qnn, qnn,
                &one,  V, idx2D(i, j, ldv), ldv, strideV,
                       workT, 0, ldt, strideT,
                &zero, W, idx2D(i, j, ldw), ldw, strideW, batch_count, workArr);
            // DZij = Vi^H Wj
            // TODO qmm?
            rocsolver_gemm(
                handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
                inc, qnn, qm,
                &one,  V, idx2D(i, i, ldv), ldv, strideV,
                       W, idx2D(i, j, ldw), ldw, strideW,
                &zero, workZ, 0, ldz, strideZ, batch_count, workArr);
            // Wj = Wj - Wi DZij
            /// TODO qmm?
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, qnn, inc,
                &negone, W, idx2D(i, i, ldv), ldw, strideW,
                         workZ, 0, ldz, strideZ,
                &one,    W, idx2D(i, j, ldw), ldw, strideW, batch_count, workArr);

            // prepare symmetric rank update
            // Z0 = A0 W
            /// TODO hemm?
            /// TODO qmm x ... x qmm?
            /// This is doing A * Wi, when need only A * Wj.
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, inc+qnn, qm,
                &one,  Acpy, 0, ldacpy, strideAcpy,
                       W, idx2D(i, i, ldw), ldw, strideW,
                &zero, workZ, 0, ldz, strideZ, batch_count, workArr);
            // DT = W^H Z0
            /// TODO qmm?
            rocsolver_gemm(
                handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
                inc+qnn, inc+qnn, qm,
                &one,  W, idx2D(i, i, ldw), ldw, strideW,
                       workZ, 0, ldz, strideZ,
                &zero, workT, 0, ldt, strideT, batch_count, workArr);
            // Z = Z0 - 0.5 V DT = Z0 - 0.5 V (W^H Z) = AVT - 0.5 V T^H V^H AVT
            /// TODO qmm?
            rocsolver_gemm(
                handle, rocblas_operation_none, rocblas_operation_none,
                qm, inc+qnn, inc+qnn,
                &neghalf, V, idx2D(i, i, ldv), ldv, strideV,
                          workT, 0, ldt, strideT,
                &one,     workZ, 0, ldz, strideZ, batch_count, workArr);

            ///j += kd;
        }

        // update trailing matrix (her2k as 2 gemms)
        inc = j - i;
        qmm = n_kd - j;
        // A = A - V Z^H
        rocsolver_gemm(
            handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
            qmm + kd, qmm + kd, kk,
            &negone, V, idx2D(i+inc-kd, i, ldv), ldv, strideV,
                     workZ, inc-kd, ldz, strideZ,
            &one,    A, shiftA + idx2D(j, j, lda), lda, strideA, batch_count, workArr);
        // A = A - Z V^H
        rocsolver_gemm(
            handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
            qmm + kd, qmm + kd, kk,
            &negone, workZ, inc-kd, ldz, strideZ,
                     V, idx2D(i+inc-kd, i, ldv), ldv, strideV,
            &one,    A, shiftA + idx2D(j, j, lda), lda, strideA, batch_count, workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

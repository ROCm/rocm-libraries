/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "laset.hpp"  // todo: just for ceildiv

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
// defined in rocauxiliary_sb2st_hb2st.hpp.
// todo: should this go in a header somewhere?
template <typename I>
__device__ rocblas_int get_v_block_index(
    I nt, I i, I j );

//------------------------------------------------------------------------------
// Sets size_* and max_parallel for the maximum number of operations to do in
// one batch.
// If batch_count > 1, max_parallel = 1.
// If batch_count = 1, max_parallel = ceildiv( nt, 2 ).
// todo: if needed, limit max_parallel to limit workspace.
//
template <bool BATCHED, typename T>
void rocsolver_ormtr_unmtr_hb2st_getMemorySize(
    const rocblas_side side,
    const rocblas_operation trans,
    const rocblas_int m,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int batch_count,
    rocblas_int* max_parallel,
    size_t* size_scalars,
    size_t* size_T,
    size_t* size_W,
    size_t* size_Z,
    size_t* size_work,
    size_t* size_workArr)
{
    *max_parallel = 1;
    *size_scalars = 0;
    *size_T = 0;
    *size_W = 0;
    *size_Z = 0;
    *size_work = 0;
    *size_workArr = 0;

    // quick return if no workspace needed
    if(m == 0 || n == 0 || kd == 0 || batch_count == 0)
        return;

    rocblas_int nz = (side == rocblas_side_left ? n : m);  // cols in Z
    rocblas_int nq = (side == rocblas_side_left ? m : n);  // rows in Q
    rocblas_int nt = ceildiv( nq - 1, kd );  // block cols in conceptual V

    // If batch_count = 1, use batched larft and gemm inside a single
    // matrix. If batch_count > 1, set max_parallel = 1 and use batching for
    // the user batch.
    if (batch_count == 1)
        max_parallel = ceildiv( nt, 2 );
    *size_Z = sizeof(T) * kd * nz * batch_count * max_parallel;
    *size_T = sizeof(T) * kd * kd * batch_count * max_parallel;
    *size_W = sizeof(T) * 2*kd * kd * batch_count * max_parallel;
    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : 0;
printf( "getMemSize1 scalars %lu, T %lu, W %lu, Z %lu, work %lu, workArr %lu\n",
        *size_scalars, *size_T, *size_W, *size_Z, *size_work, *size_workArr );

    // extra space for larft calls
    size_t w, wa;
    rocsolver_larft_getMemorySize<BATCHED, T>(
        2*kd, kd, batch_count*max_parallel, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);

printf( "getMemSize2 scalars %lu, T %lu, W %lu, Z %lu, work %lu, workArr %lu\n",
        *size_scalars, *size_T, *size_W, *size_Z, *size_work, *size_workArr );
}

//------------------------------------------------------------------------------
template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_ormtr_hb2st_argCheck(
    rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const rocblas_int m,
    const rocblas_int n,
    const rocblas_int kd,
    T V,
    const rocblas_int ldv,
    U tau,  // why U? claude put last
    T C,
    const rocblas_int ldc)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        std::cout << "side invalid: " << side << "\n";
        return rocblas_status_invalid_value;
    }

    // rocblas_operation_transpose invalid for complex;
    // rocblas_operation_conjugate_transpose ok for both real and complex.
    if(COMPLEX && trans == rocblas_operation_transpose)
    {
        std::cout << "trans invalid: " << trans << "\n";
        return rocblas_status_invalid_value;
    }

    if(trans != rocblas_operation_none
       && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
    {
        std::cout << "trans invalid: " << trans << "\n";
        return rocblas_status_invalid_value;
    }

    // 2. invalid size
    if(m < 0 || n < 0 || kd < 0 || ldv < 3*kd - 1 || ldc < m)
    {
        std::cout << "size invalid"
                   << ", m < 0: " << (m < 0)
                   << ", n < 0: " << (n < 0)
                   << ", kd < 0: " << (kd < 0)
                   << ", ldv < 3*kd - 1: " << (ldv < 3*kd - 1)
                   << ", ldc < m: " << (ldc < m)
                   << "\n";
        return rocblas_status_invalid_size;
    }

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // I would suggest quick return, then check pointers.
    // Skip pointer check if quick return.
    if (m == 0 || n == 0 || kd == 0)
        return rocblas_status_continue;

    // 3. invalid pointers
    // Usually: (m > 0 && n > 0 && kd > 0 && (!V || !tau || !C))
    if(!V || !tau || !C)
    {
        std::cout << "ptr invalid"
                  << ", V: " << V
                  << ", tau: " << tau
                  << ", C: " << C
                  << "\n";
        return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(
    rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const rocblas_int m,
    const rocblas_int n,
    const rocblas_int kd,
    U V,
    const rocblas_int shiftV,  // todo
    const rocblas_int ldv,
    const rocblas_stride strideV,
    T* tau,
    const rocblas_stride strideTau,
    U C,
    const rocblas_int shiftC,  // todo
    const rocblas_int ldc,
    const rocblas_stride strideC,
    const rocblas_int batch_count,
    const rocblas_int max_parallel,
    T* scalars,
    T* Tr, const rocblas_int ldt,
    T* W,  const rocblas_int ldw,
    T* Z,  const rocblas_int ldz,
    T* work,
    T** workArr)
{
    ROCSOLVER_ENTER("ormtr_unmtr_hb2st", "side:", side, "trans:", trans,
                    "m:", m, "n:", n, "kd:", kd,
                    "shiftV:", shiftV, "ldv:", ldv,
                    "shiftC:", shiftC, "ldc:", ldc,
                    "mp:", max_parallel, "bc:", batch_count);

    const T zero = 0;
    const T one = 1;
    const T negone = -1;

    // quick return
    if(m == 0 || n == 0 || kd == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nz = (side == rocblas_side_left ? n : m);  // cols in Z
    rocblas_int nq = (side == rocblas_side_left ? m : n);  // rows in Q
    rocblas_int nt = ceildiv( nq - 1, kd );  // block cols in conceptual V

    rocblas_stride strideTr = ldt*kd;
    rocblas_stride strideW  = ldw*kd;
    rocblas_stride strideZ  = ldz*nz;
printf( "unmtr_hb2st side %d, trans %d, m %d, n %d, kd %d, nq %d, nz %d, nt %d\n",
        int(side), int(trans), m, n, kd, nq, nz, nt );

    // k loop goes over sets of Vs that can be done in parallel.
    // j loop goes over columns within each set.
    // For instance, k = 0; j = 0, 1, 2 applies the (3) parallelograms for
    // k = 0 in the "k sets" figure, which can be done in parallel.
    // See diagram for get_v_block_index in rocauxiliary_sb2st_hb2st.hpp.
    //
    // Apply backward (right to left) or forward (left to right)?
    // hmm... is this opposite direction in larft?
    bool left = (side == rocblas_side_left);
    bool backward = left == (trans == rocblas_operation_none);
    rocblas_int k_begin, k_end, k_step;
    if (backward)
    {
        // left no-trans OR right (conj-)trans
        k_begin = nt - 1;
        k_end   = -nt;
        k_step  = -1;
    }
    else
    {
        // left (conj-)trans OR right no-trans
        k_begin = -(nt - 1);
        k_end   = nt;
        k_step  = 1;
    }
printf( "backward %d, k = %d:%d:%d\n",
        backward, k_begin, k_end, k_step );

    for (rocblas_int k = k_begin; k != k_end; k += k_step)
    {
        // i, j are block indices of the top of each conceptual V{i,j} block.
        rocblas_int j_begin = std::max( 0, k );
        rocblas_int j_end   = ceildiv( nt + k, 2 );
        printf( "k = %d, j = %d:%d\n", k, j_begin, j_end );

        rocblas_int j = j_begin;
        while (j < j_end)
        {
            // r is storage index of V{i,j} block.
            rocblas_int i = 2*j - k;
            rocblas_int r = get_v_block_index( nt, i, j );
            // old: r = i - j + j*nt - j*(j-1)/2;
            printf( "k = %d, j = %d, i = %d, r = %d\n",
                    k, j, i, r );

            // For side = left,  ii is top row of C block.
            // for side = right, ii is left col of C block.
            rocblas_int ii = i*kd + 1;

            // V block has dimensions mv-by-kv.
            rocblas_int mv = std::min( 2*kd - 1, nq - ii );
            rocblas_int kv = std::min( mv, kd );

            // Check dimensions (mv, kv) for last j in this batch. If it is
            // different, save that j for the next batch, which will be a
            // cleanup with batch_count = 1.
            rocblas_int j_last = std::min( j + max_parallel, j_end ) - 1;
            {
                rocblas_int i_last = 2*j_last - k;
                rocblas_int ii_last = i_last*kd + 1;
                rocblas_int mv_last = std::min( 2*kd - 1, nq - ii );
                rocblas_int kv_last = std::min( mv, kd );
                if (mv_last != mv || kv_last != kv)
                {
                    j_last -= 1;
                }
            }

            for (rocblas_int j_ = j; j_ <= j_last; ++j_)
            {
                // verify batch parameters.
                // todo: replace with batch calls.
                assert( i == 2*j_ - k );
                rocblas_int r_ = get_v_block_index( nt, i, j_ );
                assert( r_ == r );
                rocblas_int ii_ = i*kd + 1;
                assert( ii_ == ii );
                rocblas_int vj_ = r2*kd;
                assert( vj_ == vj );

                // jp indexes T, W, Z arrays for operations in parallel.
                // T, W are block cols; Z is block rows.
                // T = [ T0, T1, ... ]
                // W = [ W0, W1, ... ]
                // Z = [ Z0  ]
                //     [ Z1  ]
                //     [ ... ]
                rocblas_int jp = 0;  // (j_ - j) * kd;

                // Generate T, dim: (kv x kv)
                rocsolver_larft_template<T>(
                    handle, rocblas_forward_direction, rocblas_column_wise,
                    mv, kv,
                    V, r*kd*ldv, ldv, strideV,
                    &tau[ r*kd ], strideTau,
                    &Tr[ jp*ldt ], ldt, strideTr,
                    batch_count, scalars, work, workArr );

                // Rest of code is equivalent to larfb to apply a block reflector,
                // but using gemm instead of trmm.
                // W = V * op(T), dim: (mv x kv) = (mv x kv) (kv x kv)
                auto opT = backward ? rocblas_operation_none
                                    : rocblas_operation_conjugate_transpose;
                rocsolver_gemm(
                    handle, rocblas_operation_none, opT,
                    mv, kv, kv,
                    &one,  V,  r*kd*ldv, ldv, strideV,
                           Tr, jp*ldt,   ldt, strideTr,
                    &zero, W,  jp*ldw,   ldw, strideW,
                    batch_count, workArr );

                if (left)
                {
                    printf( "left\n" );
                    // Update block row Ci.
                    // Ci = op(Q) Ci = (I - V op(T) V^H) Ci
                    //    = Ci - (V op(T)) (V^H Ci)
                    //    = Ci - W Z

                    // Z = V^H Ci  (kv x n) = (mv x kv)^H (mv x n)
                    rocsolver_gemm(
                        handle,
                        rocblas_operation_conjugate_transpose,
                        rocblas_operation_none,
                        kv, n, mv,
                        &one,  V, r*kd*ldv, ldv, strideV,
                               C, ii,       ldc, strideC,
                        &zero, Z, jp,       ldz, strideZ,
                        batch_count, workArr );

                    // Ci -= W Z  (mv x n) = (mv x kv) (kv x n)
                    rocsolver_gemm(
                        handle,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        mv, n, kv,
                        &negone, W, jp*ldw, ldw, strideW,
                                 Z, jp,     ldz, strideZ,
                        &one,    C, ii,     ldc, strideC,
                        batch_count, workArr );
                }
                else // right
                {
                    printf( "right\n" );
                    // Update block col Ci.
                    // Ci = Ci op(Q) = Ci (I - V op(T) V^H)
                    //    = Ci - (Ci V) (op(T) V^H)
                    //    = Ci - Z^H W^H
                    // W = V op(T)^H, above.

                    // Z = V^H Ci^H, dim: (kv x m) = (mv x kv)^H (m x mv)^H
                    // todo: perhaps different opA, opB would be better performance,
                    // but less important since side = right is not used in heev*.
                    rocsolver_gemm(
                        handle,
                        rocblas_operation_conjugate_transpose,
                        rocblas_operation_conjugate_transpose,
                        kv, n, mv,
                        &one,  V, r*kd*ldv, ldv, strideV,
                               C, ii,       ldc, strideC,
                        &zero, Z, jp,       ldz, strideZ,
                        batch_count, workArr );

                    // Ci -= Z^H W^H, dim: (m x mv) = (kv x m)^H (mv x kv)^H
                    rocsolver_gemm(
                        handle,
                        rocblas_operation_conjugate_transpose,
                        rocblas_operation_conjugate_transpose,
                        m, mv, kv,
                        &negone, W, jp*ldw, ldw, strideW,
                                 Z, jp,     ldz, strideZ,
                        &one,    C, ii,     ldc, strideC,
                        batch_count, workArr );
                }

                // Adjust parameters to next matrix in batch.
                // todo: replace with stride in batch calls.
                i += 2;
                r += 1;
                vj += kd;
                ii += 2*kd;
            }
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

/************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_datatype2string.hpp"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
// Sets size_* and max_parallel for the maximum number of operations to do in
// one batch.
// If batch_count > 1, max_parallel = 1.
// If batch_count = 1, max_parallel = ceildiv( nt, 2 ).
// If needed, we can limit max_parallel to limit workspace memory.
//
template <bool BATCHED, typename T, typename I = rocblas_int>
void rocsolver_ormtr_unmtr_hb2st_getMemorySize(const rocblas_side side,
                                               const rocblas_operation trans,
                                               const I m,
                                               const I n,
                                               const I kd,
                                               const I batch_count,
                                               I* max_parallel,
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

    I nz = (side == rocblas_side_left ? n : m); // cols in Z
    I nq = (side == rocblas_side_left ? m : n); // rows & cols in Q
    I nt = ceildiv(nq - 1, kd); // block cols in conceptual triangular V

    // quick return if no workspace needed
    if(m == 0 || n == 0 || nq == 1 || batch_count == 0)
        return;

    // If batch_count = 1, set max_parallel > 1 and batch update block rows or
    // cols of a single matrix.
    // If batch_count > 1, set max_parallel = 1 and batch multiple matrices.
    if(batch_count == 1)
    {
        *max_parallel = ceildiv(nt, 2);
    }
    I bc = batch_count * (*max_parallel);
    bool batched = bc > 1;
    *size_Z = sizeof(T) * kd * nz * bc;
    *size_T = sizeof(T) * kd * kd * bc;
    *size_W = sizeof(T) * 2 * kd * kd * bc;
    *size_workArr = batched ? sizeof(T*) * 2 * bc : 0;

    // extra space for larft calls
    size_t w, wa;
    rocsolver_larft_getMemorySize<true, T>( // always (strided) batched
        2 * kd, kd, bc, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);
}

//------------------------------------------------------------------------------
template <bool COMPLEX, typename T, typename U, typename I = rocblas_int>
rocblas_status rocsolver_ormtr_unmtr_hb2st_argCheck(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const I m,
                                                    const I n,
                                                    const I kd,
                                                    T V,
                                                    const I ldv,
                                                    U tau,
                                                    T C,
                                                    const I ldc,
                                                    const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        return rocblas_status_invalid_value;
    }

    // rocblas_operation_transpose invalid for complex;
    // rocblas_operation_conjugate_transpose ok for both real and complex.
    if(COMPLEX && trans == rocblas_operation_transpose)
    {
        return rocblas_status_invalid_value;
    }

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
    {
        return rocblas_status_invalid_value;
    }

    // 2. invalid size
    if(m < 0 || n < 0 || kd < 1 || ldv < 2 * kd - 1 || ldc < m || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // skip pointer check if quick return
    I nq = (side == rocblas_side_left ? m : n); // rows & cols in Q
    if(m == 0 || n == 0 || nq == 1 || batch_count == 0)
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!V || !tau || !C)
    {
        return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename U, typename I = rocblas_int>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const I m,
                                                    const I n,
                                                    const I kd,
                                                    U V,
                                                    const I shiftV,
                                                    const I ldv,
                                                    rocblas_stride strideV,
                                                    T* tau,
                                                    rocblas_stride strideTau,
                                                    U C,
                                                    const I shiftC,
                                                    const I ldc,
                                                    rocblas_stride strideC,
                                                    const I batch_count,
                                                    I max_parallel,
                                                    T* scalars,
                                                    T* Tr,
                                                    T* W,
                                                    T* Z,
                                                    T* work,
                                                    T** workArr)
{
    ROCSOLVER_ENTER("ormtr_unmtr_hb2st", "side:", side, "trans:", trans, "m:", m, "n:", n,
                    "kd:", kd, "shiftV:", shiftV, "ldv:", ldv, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count, "mp:", max_parallel);

    const T zero = 0;
    const T one = 1;
    const T negone = -1;

    bool left = (side == rocblas_side_left);
    I nz = (left ? n : m); // cols in Z
    I nq = (left ? m : n); // rows & cols in Q
    I nt = ceildiv(nq - 1, kd); // block cols in conceptual triangular V

    // quick return
    if(m == 0 || n == 0 || nq == 1 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I ldt = kd;
    I ldw = 2 * kd;
    I ldz = kd;

    rocblas_stride strideT = ldt * kd;
    rocblas_stride strideW = ldw * kd;
    rocblas_stride strideZ = ldz * nz;

    I bc = batch_count;
    if(max_parallel > 1)
    {
        // max_parallel > 1 implies a non-batched call. Override the strides
        // (which are generally 0) and use batched gemm for parallel execution
        // of tasks.
        assert(bc == 1);
        strideV = ldv * kd;
        strideTau = kd;
        strideC = 2 * kd;
        // For side=right, C is strided by cols instead of rows.
        if(side == rocblas_side_right)
        {
            strideC *= ldc;
        }
    }

    // k loop goes over sets of Vs that can be done in parallel.
    // j loop goes over columns within each set.
    // For instance, k = 0; j = 0, 1, 2 applies the (3) parallelograms for
    // k = 0 in the "k sets" figure, which can be done in parallel.
    // See diagram for get_v_block_index in lib_device_helpers.hpp.
    //
    // Apply k descending (right to left) or ascending (left to right)?
    bool descend = left == (trans == rocblas_operation_none);
    I k_begin, k_end, k_step;
    if(descend)
    {
        // left no-trans OR right (conj-)trans
        k_begin = nt - 1;
        k_end = -nt;
        k_step = -1;
    }
    else
    {
        // left (conj-)trans OR right no-trans
        k_begin = -(nt - 1);
        k_end = nt;
        k_step = 1;
    }

    for(I k = k_begin; k != k_end; k += k_step)
    {
        // i, j are block indices of the top of each V{i,j} block
        // in the conceptual triangular V.
        I j_begin = std::max(I(0), k);
        I j_end = ceildiv(nt + k, I(2));

        // For given k, the j's are independent; we always go ascending.
        I j = j_begin;
        while(j < j_end)
        {
            // r is storage index of V{i,j} block.
            I i = 2 * j - k;
            I r = get_v_block_index(nt, i, j);
            I vj = r * kd;

            // For side = left,  ii is top  row of C block.
            // For side = right, ii is left col of C block.
            I ii = i * kd + 1;

            // V block has dimensions mv-by-kv.
            I mv = std::min(2 * kd - 1, nq - ii);
            I kv = std::min(mv, kd);

            // Check dimensions (mv, kv) for last j in this batch. If it is
            // different, save that j for the next batch, which will be a
            // cleanup with batch_count = 1.
            I j_last = std::min(j + max_parallel, j_end) - 1;
            {
                I i_last = 2 * j_last - k;
                I ii_last = i_last * kd + 1;
                I mv_last = std::min(2 * kd - 1, nq - ii_last);
                I kv_last = std::min(mv_last, kd);
                if(mv_last != mv || kv_last != kv)
                {
                    j_last -= 1;
                }
            }
            if(max_parallel > 1)
            {
                bc = j_last + 1 - j;
            }

            // Generate T:  kv x kv
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise, mv,
                                        kv, // opts
                                        V, vj * ldv + shiftV, ldv, strideV, // V: mv x kv
                                        &tau[vj], strideTau, // tau
                                        Tr, ldt, strideT, // T
                                        bc, scalars, work, workArr);

            // Rest of code is equivalent to larfb to apply a block reflector,
            // but using gemm instead of trmm.

            // W = V * op(T), dim: (mv x kv) = (mv x kv) (kv x kv)
            auto opT = descend ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
            rocsolver_gemm(handle, rocblas_operation_none, opT, mv, kv, kv, // opts
                           &one, V, vj * ldv + shiftV, ldv, strideV, // V
                           Tr, 0, ldt, strideT, // op(T)
                           &zero, W, 0, ldw, strideW, // W
                           bc, workArr);

            if(left)
            {
                // Update block row Ci.
                // Ci = op(Q) Ci = (I - V op(T) V^H) Ci
                //    = Ci - (V op(T)) (V^H Ci)
                //    = Ci - W Z

                // Z = V^H Ci:  (kv x n) = (mv x kv)^H (mv x n)
                rocsolver_gemm(handle, rocblas_operation_conjugate_transpose,
                               rocblas_operation_none, kv, n, mv, // opts
                               &one, V, vj * ldv + shiftV, ldv, strideV, // V^H
                               C, ii + shiftC, ldc, strideC, // C
                               &zero, Z, 0, ldz, strideZ, // Z
                               bc, workArr);

                // Ci -= W Z:  (mv x n) = (mv x kv) (kv x n)
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mv, n,
                               kv, // opts
                               &negone, W, 0, ldw, strideW, // W
                               Z, 0, ldz, strideZ, // Z
                               &one, C, ii + shiftC, ldc, strideC, // C
                               bc, workArr);
            }
            else // right
            {
                // Update block col Ci.
                // Ci = Ci op(Q) = Ci (I - V op(T) V^H)
                //    = Ci - (Ci V) (op(T) V^H)
                //    = Ci - Z^H W^H
                // W = V op(T)^H, above.

                // Z = Vr^H Ci^H:  (kv x m) = (mv x kv)^H (m x mv)^H
                rocsolver_gemm(handle, rocblas_operation_conjugate_transpose,
                               rocblas_operation_conjugate_transpose, kv, m, mv, // opts
                               &one, V, vj * ldv + shiftV, ldv, strideV, // V^H
                               C, ii * ldc + shiftC, ldc, strideC, // C^H
                               &zero, Z, 0, ldz, strideZ, // Z
                               bc, workArr);

                // Ci -= Z^H W^H:  (m x mv) = (kv x m)^H (mv x kv)^H
                rocsolver_gemm(handle, rocblas_operation_conjugate_transpose,
                               rocblas_operation_conjugate_transpose, m, mv, kv, // opts
                               &negone, Z, 0, ldz, strideZ, // Z^H
                               W, 0, ldw, strideW, // W^H
                               &one, C, ii * ldc + shiftC, ldc, strideC, // C
                               bc, workArr);
            }

            j = j_last + 1;
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

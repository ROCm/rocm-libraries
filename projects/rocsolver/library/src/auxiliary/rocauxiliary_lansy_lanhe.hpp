/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routines (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
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

#include "rocauxiliary_lantr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    LANSY and LANHE (the kernels are those of LANTR)
*************************************************************/

template <typename T, typename I, typename S>
void rocsolver_lansy_lanhe_getMemorySize(const rocsolver_norm_type norm_type,
                                         const I n,
                                         const I batch_count,
                                         size_t* size_work)
{
    rocsolver_lan_getMemorySize<LAN_KIND_SY, T, I, S>(norm_type, n, n, batch_count, size_work);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_lansy_lanhe_argCheck(rocblas_handle handle,
                                              const rocsolver_norm_type norm_type,
                                              const rocblas_fill uplo,
                                              const I n,
                                              const I lda,
                                              T A,
                                              S* norm)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
       && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !norm))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool HERM, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_lansy_lanhe_template(rocblas_handle handle,
                                              const rocsolver_norm_type norm_type,
                                              const rocblas_fill uplo,
                                              const I n,
                                              U A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              const I batch_count,
                                              S* norms,
                                              S* work)
{
    ROCSOLVER_ENTER((HERM ? "lanhe" : "lansy"), "norm_type:", norm_type, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    constexpr int KIND = HERM ? LAN_KIND_HE : LAN_KIND_SY;
    return rocsolver_lan_template<KIND, T>(handle, norm_type, uplo, rocblas_diagonal_non_unit, n, n,
                                           A, shiftA, lda, strideA, batch_count, norms, work);
}

ROCSOLVER_END_NAMESPACE

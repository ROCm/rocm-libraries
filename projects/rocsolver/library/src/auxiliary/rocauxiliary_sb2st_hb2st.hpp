/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

 #include "rocblas.hpp"
 #include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(const rocblas_int n,
                                         const rocblas_int nb,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars)
{}

template <typename T, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              const rocblas_int lda,
                                              T A,
                                              S* D,
                                              S* E,
                                              const rocblas_int batch_count = 1)
{
    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              const rocblas_int batch_count,
                                              T* scalars)
{
    ROCSOLVER_ENTER("sb2st_hb2st", "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
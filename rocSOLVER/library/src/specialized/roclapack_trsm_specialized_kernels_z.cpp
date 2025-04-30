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

#include "roclapack_trsm_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRSM_MEM(0, 0, rocblas_double_complex, rocblas_int);
INSTANTIATE_TRSM_LOWER(0, 0, rocblas_double_complex, rocblas_int, rocblas_double_complex*);
INSTANTIATE_TRSM_UPPER(0, 0, rocblas_double_complex, rocblas_int, rocblas_double_complex*);

INSTANTIATE_TRSM_MEM(0, 1, rocblas_double_complex, rocblas_int);
INSTANTIATE_TRSM_LOWER(0, 1, rocblas_double_complex, rocblas_int, rocblas_double_complex*);
INSTANTIATE_TRSM_UPPER(0, 1, rocblas_double_complex, rocblas_int, rocblas_double_complex*);

INSTANTIATE_TRSM_MEM(1, 0, rocblas_double_complex, rocblas_int);
INSTANTIATE_TRSM_LOWER(1, 0, rocblas_double_complex, rocblas_int, rocblas_double_complex* const*);
INSTANTIATE_TRSM_UPPER(1, 0, rocblas_double_complex, rocblas_int, rocblas_double_complex* const*);

#ifdef HAVE_ROCBLAS_64
// 64-bit
INSTANTIATE_TRSM_MEM(0, 0, rocblas_double_complex, int64_t);
INSTANTIATE_TRSM_LOWER(0, 0, rocblas_double_complex, int64_t, rocblas_double_complex*);
INSTANTIATE_TRSM_UPPER(0, 0, rocblas_double_complex, int64_t, rocblas_double_complex*);

INSTANTIATE_TRSM_MEM(0, 1, rocblas_double_complex, int64_t);
INSTANTIATE_TRSM_LOWER(0, 1, rocblas_double_complex, int64_t, rocblas_double_complex*);
INSTANTIATE_TRSM_UPPER(0, 1, rocblas_double_complex, int64_t, rocblas_double_complex*);

INSTANTIATE_TRSM_MEM(1, 0, rocblas_double_complex, int64_t);
INSTANTIATE_TRSM_LOWER(1, 0, rocblas_double_complex, int64_t, rocblas_double_complex* const*);
INSTANTIATE_TRSM_UPPER(1, 0, rocblas_double_complex, int64_t, rocblas_double_complex* const*);
#endif /* HAVE_ROCBLAS_64 */

ROCSOLVER_END_NAMESPACE

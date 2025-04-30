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

#include "roclapack_getf2_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_PANEL(float, rocblas_int, rocblas_int, float*);
INSTANTIATE_GETF2_PANEL(float, rocblas_int, rocblas_int, float* const*);

INSTANTIATE_GETF2_SCALE_UPDATE(float, rocblas_int, float*);
INSTANTIATE_GETF2_SCALE_UPDATE(float, rocblas_int, float* const*);

#ifdef HAVE_ROCBLAS_64
// 64-bit APIs
INSTANTIATE_GETF2_PANEL(float, int64_t, int64_t, float*);
INSTANTIATE_GETF2_PANEL(float, int64_t, int64_t, float* const*);

INSTANTIATE_GETF2_SCALE_UPDATE(float, int64_t, float*);
INSTANTIATE_GETF2_SCALE_UPDATE(float, int64_t, float* const*);
#endif /* HAVE_ROCBLAS_64 */

ROCSOLVER_END_NAMESPACE

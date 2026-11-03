/* **************************************************************************
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

#include "roclapack_gebak.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gebak_impl(rocblas_handle handle,
                                    const rocsolver_balance job,
                                    const rocblas_side side,
                                    const I n,
                                    const I* ilo,
                                    const I* ihi,
                                    const S* scale,
                                    const I m,
                                    U V,
                                    const I ldv)
try
{
    ROCSOLVER_ENTER_TOP("gebak", "--job", job, "--side", side, "-n", n, "-m", m, "--ldv", ldv);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gebak_argCheck(handle, job, side, n, ilo, ihi, scale, m, V, ldv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftV = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideS = 0;
    rocblas_stride strideV = 0;
    I batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_gebak_template<false, false, T>(handle, job, side, n, ilo, ihi, scale, strideS,
                                                     m, V, shiftV, ldv, strideV, batch_count);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgebak(rocblas_handle handle,
                                const rocsolver_balance job,
                                const rocblas_side side,
                                const rocblas_int n,
                                const rocblas_int* ilo,
                                const rocblas_int* ihi,
                                const float* scale,
                                const rocblas_int m,
                                float* V,
                                const rocblas_int ldv)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebak_impl<float>(handle, job, side, n, ilo, ihi, scale, m, V, ldv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgebak(rocblas_handle handle,
                                const rocsolver_balance job,
                                const rocblas_side side,
                                const rocblas_int n,
                                const rocblas_int* ilo,
                                const rocblas_int* ihi,
                                const double* scale,
                                const rocblas_int m,
                                double* V,
                                const rocblas_int ldv)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebak_impl<double>(handle, job, side, n, ilo, ihi, scale, m, V, ldv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgebak(rocblas_handle handle,
                                const rocsolver_balance job,
                                const rocblas_side side,
                                const rocblas_int n,
                                const rocblas_int* ilo,
                                const rocblas_int* ihi,
                                const float* scale,
                                const rocblas_int m,
                                rocblas_float_complex* V,
                                const rocblas_int ldv)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebak_impl<rocblas_float_complex>(handle, job, side, n, ilo, ihi,
                                                                  scale, m, V, ldv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgebak(rocblas_handle handle,
                                const rocsolver_balance job,
                                const rocblas_side side,
                                const rocblas_int n,
                                const rocblas_int* ilo,
                                const rocblas_int* ihi,
                                const double* scale,
                                const rocblas_int m,
                                rocblas_double_complex* V,
                                const rocblas_int ldv)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebak_impl<rocblas_double_complex>(handle, job, side, n, ilo, ihi,
                                                                   scale, m, V, ldv);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

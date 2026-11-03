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

#include "roclapack_trevc3.hpp"

#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_trevc3_strided_batched_impl(rocblas_handle handle,
                                                     const rocblas_side side,
                                                     const rocsolver_eigenvectors howmny,
                                                     const I n,
                                                     U A,
                                                     const I ldt,
                                                     const rocblas_stride strideT,
                                                     U VL,
                                                     const I ldvl,
                                                     const rocblas_stride strideVL,
                                                     U VR,
                                                     const I ldvr,
                                                     const rocblas_stride strideVR,
                                                     const I batch_count)
try
{
    ROCSOLVER_ENTER_TOP("trevc3_strided_batched", "--side", side, "--howmny", howmny, "-n", n,
                        "--ldt", ldt, "--strideT", strideT, "--ldvl", ldvl, "--strideVL", strideVL,
                        "--ldvr", ldvr, "--strideVR", strideVR, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trevc3_argCheck(handle, side, howmny, n, ldt, ldvl, ldvr, A, VL,
                                                  VR, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftT = 0;
    rocblas_stride shiftVL = 0;
    rocblas_stride shiftVR = 0;

    // memory workspace sizes:
    // size of the vectors of a block of eigenvectors and of their back-transformation
    size_t size_X, size_tmp;
    // size of the right-hand sides of the diagonal-block solves
    size_t size_R;
    // size of the bounds on the row and column sums of T
    size_t size_tmax;
    // size of the array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_trevc3_getMemorySize<false, T>(side, howmny, n, batch_count, &size_X, &size_tmp,
                                             &size_R, &size_tmax, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_X, size_tmp, size_R, size_tmax,
                                                      size_workArr);

    // memory workspace allocation
    void *X, *tmp, *R, *tmax, *workArr;
    rocblas_device_malloc mem(handle, size_X, size_tmp, size_R, size_tmax, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    X = mem[0];
    tmp = mem[1];
    R = mem[2];
    tmax = mem[3];
    workArr = mem[4];

    // execution
    return rocsolver_trevc3_template<false, true, T>(
        handle, side, howmny, n, A, shiftT, ldt, strideT, VL, shiftVL, ldvl, strideVL, VR, shiftVR,
        ldvr, strideVR, batch_count, (T*)X, (T*)tmp, (T*)R, tmax, (T**)workArr);
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

rocblas_status rocsolver_ctrevc3_strided_batched(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocsolver_eigenvectors howmny,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* T,
                                                 const rocblas_int ldt,
                                                 const rocblas_stride strideT,
                                                 rocblas_float_complex* VL,
                                                 const rocblas_int ldvl,
                                                 const rocblas_stride strideVL,
                                                 rocblas_float_complex* VR,
                                                 const rocblas_int ldvr,
                                                 const rocblas_stride strideVR,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_TREVC)
    return rocsolver::rocsolver_trevc3_strided_batched_impl<rocblas_float_complex>(
        handle, side, howmny, n, T, ldt, strideT, VL, ldvl, strideVL, VR, ldvr, strideVR,
        batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ztrevc3_strided_batched(rocblas_handle handle,
                                                 const rocblas_side side,
                                                 const rocsolver_eigenvectors howmny,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* T,
                                                 const rocblas_int ldt,
                                                 const rocblas_stride strideT,
                                                 rocblas_double_complex* VL,
                                                 const rocblas_int ldvl,
                                                 const rocblas_stride strideVL,
                                                 rocblas_double_complex* VR,
                                                 const rocblas_int ldvr,
                                                 const rocblas_stride strideVR,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_TREVC)
    return rocsolver::rocsolver_trevc3_strided_batched_impl<rocblas_double_complex>(
        handle, side, howmny, n, T, ldt, strideT, VL, ldvl, strideVL, VR, ldvr, strideVR,
        batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

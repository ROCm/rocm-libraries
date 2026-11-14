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

#include "roclapack_geev.hpp"

#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_geev_impl(rocblas_handle handle,
                                   const rocblas_evect jobvl,
                                   const rocblas_evect jobvr,
                                   const I n,
                                   U A,
                                   const I lda,
                                   T* W,
                                   U VL,
                                   const I ldvl,
                                   U VR,
                                   const I ldvr,
                                   I* info)
try
{
    ROCSOLVER_ENTER_TOP("geev", "--jobvl", jobvl, "--jobvr", jobvr, "-n", n, "--lda", lda, "--ldvl",
                        ldvl, "--ldvr", ldvr);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_geev_argCheck(handle, jobvl, jobvr, n, lda, ldvl, ldvr, A, W, VL, VR, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftVL = 0;
    rocblas_stride shiftVR = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideVL = 0;
    rocblas_stride strideVR = 0;
    I batch_count = 1;

    // memory workspace sizes (see rocsolver_geev_getMemorySize)
    size_t size_scalars, size_work1, size_work2, size_work3, size_work4, size_work5, size_work6;
    size_t size_tau, size_scale, size_iloihi, size_anrm, size_ptrs;
    rocsolver_geev_getMemorySize<false, T>(handle, jobvl, jobvr, n, batch_count, &size_scalars,
                                           &size_work1, &size_work2, &size_work3, &size_work4,
                                           &size_work5, &size_work6, &size_tau, &size_scale,
                                           &size_iloihi, &size_anrm, &size_ptrs);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_work5,
            size_work6, size_tau, size_scale, size_iloihi, size_anrm, size_ptrs);

    // memory workspace allocation
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6, size_tau, size_scale, size_iloihi, size_anrm,
                              size_ptrs);
    if(!mem)
        return rocblas_status_memory_error;

    void* scalars = mem[0];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_geev_template<false, false, T>(
        handle, jobvl, jobvr, n, A, shiftA, lda, strideA, W, strideW, VL, shiftVL, ldvl, strideVL,
        VR, shiftVR, ldvr, strideVR, info, batch_count, (T*)scalars, mem[1], mem[2], mem[3], mem[4],
        mem[5], mem[6], (T*)mem[7], mem[8], (I*)mem[9], mem[10], (T**)mem[11]);
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

rocblas_status rocsolver_cgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               rocblas_float_complex* W,
                               rocblas_float_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_float_complex* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
#if defined(ROCSOLVER_ENABLE_GEEV)
    return rocsolver::rocsolver_geev_impl<rocblas_float_complex>(handle, jobvl, jobvr, n, A, lda, W,
                                                                 VL, ldvl, VR, ldvr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               rocblas_double_complex* W,
                               rocblas_double_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_double_complex* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
#if defined(ROCSOLVER_ENABLE_GEEV)
    return rocsolver::rocsolver_geev_impl<rocblas_double_complex>(handle, jobvl, jobvr, n, A, lda,
                                                                  W, VL, ldvl, VR, ldvr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

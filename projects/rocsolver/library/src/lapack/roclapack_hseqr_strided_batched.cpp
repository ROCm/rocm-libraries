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

#include "exceptions.hpp"
#include "roclapack_hseqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_hseqr_strided_batched_impl(rocblas_handle handle,
                                                    const rocsolver_schur_job job,
                                                    const rocsolver_schur_vectors compz,
                                                    const I n,
                                                    const I* ilo,
                                                    const I* ihi,
                                                    U H,
                                                    const I ldh,
                                                    const rocblas_stride strideH,
                                                    T* W,
                                                    const rocblas_stride strideW,
                                                    U Z,
                                                    const I ldz,
                                                    const rocblas_stride strideZ,
                                                    I* info,
                                                    const I batch_count)
try
{
    ROCSOLVER_ENTER_TOP("hseqr_strided_batched", "--job", job, "--compz", compz, "-n", n, "--ldh",
                        ldh, "--strideH", strideH, "--strideW", strideW, "--ldz", ldz, "--strideZ",
                        strideZ, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_hseqr_argCheck(handle, job, compz, n, ilo, ihi, ldh, ldz, H, W, Z,
                                                 info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftH = 0;
    rocblas_stride shiftZ = 0;

    // memory workspace sizes:
    // sizes of the integer and scalar workspaces of the multishift algorithm (status
    // arrays, flags, grid barrier counters, reflections of the sweep and band copy)
    size_t size_work, size_workT;
    rocsolver_hseqr_getMemorySize<T>(n, batch_count, &size_work, &size_workT);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_workT);

    // memory workspace allocation
    void *work, *workT;
    rocblas_device_malloc mem(handle, size_work, size_workT);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    workT = mem[1];

    // execution
    return rocsolver_hseqr_template<false, true, T>(handle, job, compz, n, ilo, ihi, H, shiftH, ldh,
                                                    strideH, W, strideW, Z, shiftZ, ldz, strideZ,
                                                    info, batch_count, (I*)work, (T*)workT);
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

rocblas_status rocsolver_chseqr_strided_batched(rocblas_handle handle,
                                                const rocsolver_schur_job job,
                                                const rocsolver_schur_vectors compz,
                                                const rocblas_int n,
                                                const rocblas_int* ilo,
                                                const rocblas_int* ihi,
                                                rocblas_float_complex* H,
                                                const rocblas_int ldh,
                                                const rocblas_stride strideH,
                                                rocblas_float_complex* W,
                                                const rocblas_stride strideW,
                                                rocblas_float_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HSEQR)
    return rocsolver::rocsolver_hseqr_strided_batched_impl<rocblas_float_complex>(
        handle, job, compz, n, ilo, ihi, H, ldh, strideH, W, strideW, Z, ldz, strideZ, info,
        batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zhseqr_strided_batched(rocblas_handle handle,
                                                const rocsolver_schur_job job,
                                                const rocsolver_schur_vectors compz,
                                                const rocblas_int n,
                                                const rocblas_int* ilo,
                                                const rocblas_int* ihi,
                                                rocblas_double_complex* H,
                                                const rocblas_int ldh,
                                                const rocblas_stride strideH,
                                                rocblas_double_complex* W,
                                                const rocblas_stride strideW,
                                                rocblas_double_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HSEQR)
    return rocsolver::rocsolver_hseqr_strided_batched_impl<rocblas_double_complex>(
        handle, job, compz, n, ilo, ihi, H, ldh, strideH, W, strideW, Z, ldz, strideZ, info,
        batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

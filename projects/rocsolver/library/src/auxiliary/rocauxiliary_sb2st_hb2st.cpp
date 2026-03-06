/* **************************************************************************
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

#include "rocauxiliary_sb2st_hb2st.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// todo: what is difference between T, S, U?
// I gather that for complex, T=complex, S=real, U=complex*.
// Why not just use T* instead of U?

template <typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_impl(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    U Aband,
    const rocblas_int ldab,
    S* D,
    S* E,
    U V,
    const rocblas_int ldv )
{
    ROCSOLVER_ENTER_TOP( "sb2st_hb2st", "-n", n, "--kd", kd, "--ldab", ldab, "--ldv", ldv );

    if (! handle)
        return rocblas_status_invalid_handle;

    // argument checking
    // todo: why is argCheck not in same order as routine itself?
    rocblas_status st = rocsolver_sb2st_hb2st_argCheck(
        handle, n, kd, ldab, ldv, Aband, D, E, V );
        // handle, n, kd, Aband, ldab, D, E, V, ldv  // e.g.
    if (st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideV = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;
    rocsolver_sb2st_hb2st_getMemorySize<false, T, S>(
        n, kd, batch_count, &size_work );
    assert( size_work == 0 );

    if (rocblas_is_device_memory_size_query( handle ) )
        return rocblas_set_optimal_device_memory_size( handle, size_work );

    // memory workspace allocation
    /*
    void* work;
    rocblas_device_malloc mem( handle, size_work );

    if (! mem)
        return rocblas_status_memory_error;

    work = mem[0];
    */

    // todo: for what matrices do we need shift and when not?
    // todo: if there is no workspace, do we still put it for consistency and
    // future compatability?
    // execution
    return rocsolver_sb2st_hb2st_template<false, false, T>(
        handle, n, kd,
        Aband, shiftA, ldab, strideA,
        V, ldv, strideV,
        D, strideD,
        E, strideE,
        batch_count );
}

ROCSOLVER_END_NAMESPACE

/*
  * ===========================================================================
  *    C wrapper
  * ===========================================================================
  */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssb2st(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    float* Aband,
    const rocblas_int ldab,
    float* D,
    float* E,
    float* V,
    const rocblas_int ldv )
{
    return rocsolver::rocsolver_sb2st_hb2st_impl<float>(
        handle, n, kd, Aband, ldab, D, E, V, ldv );
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsb2st(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    double* Aband,
    const rocblas_int ldab,
    double* D,
    double* E,
    double* V,
    const rocblas_int ldv )
{
    return rocsolver::rocsolver_sb2st_hb2st_impl<double>(
        handle, n, kd, Aband, ldab, D, E, V, ldv );
}

ROCSOLVER_EXPORT rocblas_status rocsolver_chb2st(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    rocblas_float_complex* Aband,
    const rocblas_int ldab,
    float* D,
    float* E,
    rocblas_float_complex* V,
    const rocblas_int ldv )
{
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_float_complex>(
        handle, n, kd, Aband, ldab, D, E, V, ldv );
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhb2st(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    rocblas_double_complex* Aband,
    const rocblas_int ldab,
    double* D,
    double* E,
    rocblas_double_complex* V,
    const rocblas_int ldv )
{
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_double_complex>(
        handle, n, kd, Aband, ldab, D, E, V, ldv );
}

} // extern C

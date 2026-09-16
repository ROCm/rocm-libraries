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

#include "rocauxiliary_orghr_unghr.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_orghr_unghr_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int ilo,
                                          const rocblas_int ihi,
                                          T* A,
                                          const rocblas_int lda,
                                          T* tau)
try
{
    const char* name = (!rocblas_is_complex<T> ? "orghr" : "unghr");
    ROCSOLVER_ENTER_TOP(name, "-n", n, "--ilo", ilo, "--ihi", ihi, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_orghr_argCheck(handle, n, ilo, ihi, lda, A, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORGQx/UNGQx and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orghr_unghr_getMemorySize<false, T>(n, ilo, ihi, batch_count, &size_scalars, &size_work,
                                                  &size_Abyx_tmptr, &size_trfact, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work,
                                                      size_Abyx_tmptr, size_trfact, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *Abyx_tmptr, *trfact, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_Abyx_tmptr, size_trfact,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    Abyx_tmptr = mem[2];
    trfact = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_orghr_unghr_template<false, false, T>(
        handle, n, ilo, ihi, A, shiftA, lda, strideA, tau, strideP, batch_count, (T*)scalars,
        (T*)work, (T*)Abyx_tmptr, (T*)trfact, (T**)workArr);
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

/*! @{
    \brief ORGHR generates an n-by-n orthogonal matrix Q.

    \details
    Q is defined as the product of ``ihi - ilo`` Householder reflectors of order ``n``:

    \f[
        Q = H(\text{ilo}) H(\text{ilo}+1) \cdots H(\text{ihi}-1)
    \f]

    The Householder matrices \f$H(i)\f$ are never stored. They are computed from the
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{tau}[i]\f$, as
    returned by \ref rocsolver_sgehrd "GEHRD" in its arguments ``A`` and ``tau``.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.
                The order of the matrix Q.
    @param[in]
    ilo         rocblas_int. 1 <= ilo <= ihi.
                The lower bound (1-based) of the active submatrix.
    @param[in]
    ihi         rocblas_int. ilo <= ihi <= n.
                The upper bound (1-based) of the active submatrix.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.
                On entry, the Householder vectors as returned by \ref rocsolver_sgehrd "GEHRD".
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= n.
                The leading dimension of A.
    @param[in]
    tau        pointer to type. Array on the GPU of at least ihi - ilo scalars.
                The Householder scalars as returned by \ref rocsolver_sgehrd "GEHRD" for reflectors
                H(ilo) through H(ihi-1).
    ********************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_sorghr(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int ilo,
                                                 const rocblas_int ihi,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* tau)
{
#if defined(ROCSOLVER_ENABLE_XXGHR)
    return rocsolver::rocsolver_orghr_unghr_impl<float>(handle, n, ilo, ihi, A, lda, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dorghr(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int ilo,
                                                 const rocblas_int ihi,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* tau)
{
#if defined(ROCSOLVER_ENABLE_XXGHR)
    return rocsolver::rocsolver_orghr_unghr_impl<double>(handle, n, ilo, ihi, A, lda, tau);
#else
    return rocblas_status_not_implemented;
#endif
}
//! @}

/*! @{
    \brief UNGHR generates an n-by-n unitary matrix Q.

    \details
    Q is defined as the product of ``ihi - ilo`` Householder reflectors of order ``n``:

    \f[
        Q = H(\text{ilo}) H(\text{ilo}+1) \cdots H(\text{ihi}-1)
    \f]

    The Householder matrices \f$H(i)\f$ are never stored. They are computed from the
    corresponding Householder vectors \f$v_i\f$ and scalars \f$\text{tau}[i]\f$, as
    returned by \ref rocsolver_sgehrd "GEHRD" in its arguments ``A`` and ``tau``.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.
                The order of the matrix Q.
    @param[in]
    ilo         rocblas_int. 1 <= ilo <= ihi.
                The lower bound (1-based) of the active submatrix.
    @param[in]
    ihi         rocblas_int. ilo <= ihi <= n.
                The upper bound (1-based) of the active submatrix.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.
                On entry, the Householder vectors as returned by \ref rocsolver_sgehrd "GEHRD".
                On exit, the computed matrix Q.
    @param[in]
    lda         rocblas_int. lda >= n.
                The leading dimension of A.
    @param[in]
    tau        pointer to type. Array on the GPU of at least ihi - ilo scalars.
                The Householder scalars as returned by \ref rocsolver_sgehrd "GEHRD" for reflectors
                H(ilo) through H(ihi-1).
    ********************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_cunghr(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int ilo,
                                                 const rocblas_int ihi,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_XXGHR)
    return rocsolver::rocsolver_orghr_unghr_impl<rocblas_float_complex>(handle, n, ilo, ihi, A, lda,
                                                                        tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zunghr(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int ilo,
                                                 const rocblas_int ihi,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_XXGHR)
    return rocsolver::rocsolver_orghr_unghr_impl<rocblas_double_complex>(handle, n, ilo, ihi, A,
                                                                         lda, tau);
#else
    return rocblas_status_not_implemented;
#endif
}
//! @}

} // extern C

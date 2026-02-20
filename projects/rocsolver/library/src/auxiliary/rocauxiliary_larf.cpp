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

#include "rocauxiliary_larf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_larf_impl(rocblas_handle handle,
                                   const rocblas_side side,
                                   const I m,
                                   const I n,
                                   T* x,
                                   const I incx,
                                   const T* alpha,
                                   T* A,
                                   const I lda)
{
    ROCSOLVER_ENTER_TOP("larf", "--side", side, "-m", m, "-n", n, "--incx", incx, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_larf_argCheck(handle, side, m, n, lda, incx, x, A, alpha);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    rocblas_stride stridea = 0;
    rocblas_stride stridep = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size for temporary results in generation of Householder matrix
    size_t size_Abyx;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_larf_getMemorySize<false, T>(side, m, n, batch_count, &size_scalars, &size_Abyx,
                                           &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_Abyx, size_workArr);

    // memory workspace allocation
    void *scalars, *Abyx, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_Abyx, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    Abyx = mem[1];
    workArr = mem[2];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_larf_template<T>(handle, side, m, n, x, shiftx, incx, stridex, alpha, stridep,
                                      A, shiftA, lda, stridea, batch_count, (T*)scalars, (T*)Abyx,
                                      (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               float* x,
                               const rocblas_int incx,
                               const float* alpha,
                               float* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_larf_impl<float>(handle, side, m, n, x, incx, alpha, A, lda);
}

rocblas_status rocsolver_dlarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               double* x,
                               const rocblas_int incx,
                               const double* alpha,
                               double* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_larf_impl<double>(handle, side, m, n, x, incx, alpha, A, lda);
}

rocblas_status rocsolver_clarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               rocblas_float_complex* x,
                               const rocblas_int incx,
                               const rocblas_float_complex* alpha,
                               rocblas_float_complex* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_larf_impl<rocblas_float_complex>(handle, side, m, n, x, incx, alpha,
                                                                 A, lda);
}

rocblas_status rocsolver_zlarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               rocblas_double_complex* x,
                               const rocblas_int incx,
                               const rocblas_double_complex* alpha,
                               rocblas_double_complex* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_larf_impl<rocblas_double_complex>(handle, side, m, n, x, incx,
                                                                  alpha, A, lda);
}

rocblas_status rocsolver_slarf_64(rocblas_handle handle,
                                  const rocblas_side side,
                                  const int64_t m,
                                  const int64_t n,
                                  float* x,
                                  const int64_t incx,
                                  const float* alpha,
                                  float* A,
                                  const int64_t lda)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larf_impl<float>(handle, side, m, n, x, incx, alpha, A, lda);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dlarf_64(rocblas_handle handle,
                                  const rocblas_side side,
                                  const int64_t m,
                                  const int64_t n,
                                  double* x,
                                  const int64_t incx,
                                  const double* alpha,
                                  double* A,
                                  const int64_t lda)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larf_impl<double>(handle, side, m, n, x, incx, alpha, A, lda);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_clarf_64(rocblas_handle handle,
                                  const rocblas_side side,
                                  const int64_t m,
                                  const int64_t n,
                                  rocblas_float_complex* x,
                                  const int64_t incx,
                                  const rocblas_float_complex* alpha,
                                  rocblas_float_complex* A,
                                  const int64_t lda)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larf_impl<rocblas_float_complex>(handle, side, m, n, x, incx, alpha,
                                                                 A, lda);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlarf_64(rocblas_handle handle,
                                  const rocblas_side side,
                                  const int64_t m,
                                  const int64_t n,
                                  rocblas_double_complex* x,
                                  const int64_t incx,
                                  const rocblas_double_complex* alpha,
                                  rocblas_double_complex* A,
                                  const int64_t lda)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larf_impl<rocblas_double_complex>(handle, side, m, n, x, incx,
                                                                  alpha, A, lda);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

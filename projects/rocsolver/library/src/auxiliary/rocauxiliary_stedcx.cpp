/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_stedcx.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    stedcx is not intended for inclusion in the public API. It
 *    exists to assist in debugging syevdx and to keep the code clean.
 * ===========================================================================
 */

template <typename T, typename S>
rocblas_status rocsolver_stedcx_impl(rocblas_handle handle,
                                     const rocblas_evect evect,
                                     const rocblas_erange erange,
                                     const rocblas_int n,
                                     const S vl,
                                     const S vu,
                                     const rocblas_int il,
                                     const rocblas_int iu,
                                     S* D,
                                     S* E,
                                     rocblas_int* nev,
                                     S* W,
                                     T* C,
                                     const rocblas_int ldc,
                                     rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stedcx", "--evect", evect, "--erange", erange, "-n", n, "--vl", vl, "--vu",
                        vu, "--il", il, "--iu", iu, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_stedcx_argCheck(handle, evect, erange, n, vl, vu, il, iu, D, E,
                                                  nev, W, C, ldc, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideC = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lasrt/steqr stack workspace
    size_t size_work_stack, size_work_steqr;
    // size for temporary computations
    size_t size_tempvect, size_tempgemm;
    // size for pointers to workspace (batched case)
    size_t size_workArr;
    // size for vector with positions of split blocks
    size_t size_splits;
    // size for temporary diagonal and z vectors.
    size_t size_tmpz;
    rocsolver_stedcx_getMemorySize<false, T, S>(evect, n, batch_count, &size_work_stack,
                                                &size_work_steqr, &size_tempvect, &size_tempgemm,
                                                &size_tmpz, &size_splits, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_stack, size_work_steqr,
                                                      size_tempvect, size_tempgemm, size_tmpz,
                                                      size_splits, size_workArr);

    // memory workspace allocation
    void *work_stack, *work_steqr, *tempvect, *tempgemm, *tmpz, *splits, *workArr;
    rocblas_device_malloc mem(handle, size_work_stack, size_work_steqr, size_tempvect,
                              size_tempgemm, size_tmpz, size_splits, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    work_stack = mem[0];
    work_steqr = mem[1];
    tempvect = mem[2];
    tempgemm = mem[3];
    tmpz = mem[4];
    splits = mem[5];
    workArr = mem[6];

    // execution
    return rocsolver_stedcx_template<false, false, T>(
        handle, evect, erange, n, vl, vu, il, iu, D, strideD, E, strideE, nev, W, strideW, C,
        shiftC, ldc, strideC, info, batch_count, (S*)work_stack, (S*)work_steqr, (S*)tempvect,
        (S*)tempgemm, (S*)tmpz, (rocblas_int*)splits, (S**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sstedcx(rocblas_handle handle,
                                                  const rocblas_evect evect,
                                                  const rocblas_erange erange,
                                                  const rocblas_int n,
                                                  const float vl,
                                                  const float vu,
                                                  const rocblas_int il,
                                                  const rocblas_int iu,
                                                  float* D,
                                                  float* E,
                                                  rocblas_int* nev,
                                                  float* W,
                                                  float* C,
                                                  const rocblas_int ldc,
                                                  rocblas_int* info)
{
    return rocsolver::rocsolver_stedcx_impl<float>(handle, evect, erange, n, vl, vu, il, iu, D, E,
                                                   nev, W, C, ldc, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dstedcx(rocblas_handle handle,
                                                  const rocblas_evect evect,
                                                  const rocblas_erange erange,
                                                  const rocblas_int n,
                                                  const double vl,
                                                  const double vu,
                                                  const rocblas_int il,
                                                  const rocblas_int iu,
                                                  double* D,
                                                  double* E,
                                                  rocblas_int* nev,
                                                  double* W,
                                                  double* C,
                                                  const rocblas_int ldc,
                                                  rocblas_int* info)
{
    return rocsolver::rocsolver_stedcx_impl<double>(handle, evect, erange, n, vl, vu, il, iu, D, E,
                                                    nev, W, C, ldc, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cstedcx(rocblas_handle handle,
                                                  const rocblas_evect evect,
                                                  const rocblas_erange erange,
                                                  const rocblas_int n,
                                                  const float vl,
                                                  const float vu,
                                                  const rocblas_int il,
                                                  const rocblas_int iu,
                                                  float* D,
                                                  float* E,
                                                  rocblas_int* nev,
                                                  float* W,
                                                  rocblas_float_complex* C,
                                                  const rocblas_int ldc,
                                                  rocblas_int* info)
{
    return rocsolver::rocsolver_stedcx_impl<rocblas_float_complex>(
        handle, evect, erange, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zstedcx(rocblas_handle handle,
                                                  const rocblas_evect evect,
                                                  const rocblas_erange erange,
                                                  const rocblas_int n,
                                                  const double vl,
                                                  const double vu,
                                                  const rocblas_int il,
                                                  const rocblas_int iu,
                                                  double* D,
                                                  double* E,
                                                  rocblas_int* nev,
                                                  double* W,
                                                  rocblas_double_complex* C,
                                                  const rocblas_int ldc,
                                                  rocblas_int* info)
{
    return rocsolver::rocsolver_stedcx_impl<rocblas_double_complex>(
        handle, evect, erange, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}

} // extern C

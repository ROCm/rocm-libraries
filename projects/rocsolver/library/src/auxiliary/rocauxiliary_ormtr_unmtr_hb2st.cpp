/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_ormtr_unmtr_hb2st.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_hb2st_impl(rocblas_handle handle,
                                                const rocblas_side side,
                                                const rocblas_operation trans,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                const rocblas_int kd,
                                                T* V,
                                                const rocblas_int ldv,
                                                T* tau,
                                                T* C,
                                                const rocblas_int ldc)
{
    const char* name = (!rocblas_is_complex<T> ? "ormtr_hb2st" : "unmtr_hb2st");
    ROCSOLVER_ENTER_TOP(name, "--side", side, "--trans", trans, "-m", m, "-n", n, "--kd", kd,
                        "--ldv", ldv, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_ormtr_hb2st_argCheck<COMPLEX>(handle, side, trans, m, n, kd, ldv,
                                                                ldc, V, C, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftV = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_int strideV = 0;
    rocblas_int strideT = 0;
    rocblas_int strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    bool optim_mem;
    size_t size_scalars;
    size_t size_work, size_work2, size_work3, size_work4;
    size_t size_workArr;
    rocsolver_ormtr_unmtr_hb2st_getMemorySize<false, false, T>(
        side, trans, m, n, kd, batch_count, &size_scalars, &size_work, &size_work2, &size_work3,
        &size_work4, &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_work2,
                                                      size_work3, size_work4, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *work2, *work3, *work4, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_work2, size_work3, size_work4,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    workArr = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_ormtr_unmtr_hb2st_template<false, false, T>(
        handle, side, trans, m, n, kd, V, shiftV, ldv, strideV, tau, strideT, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)work, work2, work3, work4, (T**)workArr, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sormtr_hb2st(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_operation trans,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int kd,
                                      float* V,
                                      const rocblas_int ldv,
                                      float* tau,
                                      float* C,
                                      const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_unmtr_hb2st_impl<float>(handle, side, trans, m, n, kd, V,
                                                              ldv, tau, C, ldc);
}

rocblas_status rocsolver_dormtr_hb2st(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_operation trans,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int kd,
                                      double* V,
                                      const rocblas_int ldv,
                                      double* tau,
                                      double* C,
                                      const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_unmtr_hb2st_impl<double>(handle, side, trans, m, n, kd, V,
                                                               ldv, tau, C, ldc);
}

rocblas_status rocsolver_cunmtr_hb2st(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_operation trans,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int kd,
                                      rocblas_float_complex* V,
                                      const rocblas_int ldv,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* C,
                                      const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_unmtr_hb2st_impl<rocblas_float_complex>(
        handle, side, trans, m, n, kd, V, ldv, tau, C, ldc);
}

rocblas_status rocsolver_zunmtr_hb2st(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_operation trans,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int kd,
                                      rocblas_double_complex* V,
                                      const rocblas_int ldv,
                                      rocblas_double_complex* tau,
                                      rocblas_double_complex* C,
                                      const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_unmtr_hb2st_impl<rocblas_double_complex>(
        handle, side, trans, m, n, kd, V, ldv, tau, C, ldc);
}

} // extern C

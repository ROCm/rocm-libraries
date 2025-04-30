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

#include "roclapack_getf2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_getf2_impl(rocblas_handle handle,
                                    const I m,
                                    const I n,
                                    U A,
                                    const I lda,
                                    I* ipiv,
                                    I* info,
                                    const bool pivot)
{
    const char* name = (pivot ? "getf2" : "getf2_npvt");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, pivot);
    if(st != rocblas_status_continue)
        return st;

    // using unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftP = 0;

    // normal (non-batched non-strided) execution
    I inca = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // sizes to store pivots in intermediate computations
    size_t size_pivotval;
    size_t size_pivotidx;
    rocsolver_getf2_getMemorySize<false, T>(m, n, pivot, batch_count, &size_scalars, &size_pivotval,
                                            &size_pivotidx);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_pivotval,
                                                      size_pivotidx);

    // memory workspace allocation
    void *scalars, *pivotidx, *pivotval;
    rocblas_device_malloc mem(handle, size_scalars, size_pivotval, size_pivotidx);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    pivotval = mem[1];
    pivotidx = mem[2];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_getf2_template<false, T>(handle, m, n, A, shiftA, inca, lda, strideA, ipiv,
                                              shiftP, strideP, info, batch_count, (T*)scalars,
                                              (T*)pivotval, (I*)pivotidx, pivot);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetf2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getf2_impl<float>(handle, m, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_dgetf2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getf2_impl<double>(handle, m, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_cgetf2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getf2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info,
                                                                  true);
}

rocblas_status rocsolver_zgetf2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getf2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info,
                                                                   true);
}

rocblas_status rocsolver_sgetf2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getf2_impl<float>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetf2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getf2_impl<double>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetf2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getf2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info,
                                                                  true);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetf2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getf2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info,
                                                                   true);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_sgetf2_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<float>(handle, m, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_dgetf2_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     double* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<double>(handle, m, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_cgetf2_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     rocblas_float_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info,
                                                                  false);
}

rocblas_status rocsolver_zgetf2_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     rocblas_double_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info,
                                                                   false);
}

rocblas_status rocsolver_sgetf2_npvt_64(rocblas_handle handle,
                                        const int64_t m,
                                        const int64_t n,
                                        float* A,
                                        const int64_t lda,
                                        int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<float>(handle, m, n, A, lda, ipiv, info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetf2_npvt_64(rocblas_handle handle,
                                        const int64_t m,
                                        const int64_t n,
                                        double* A,
                                        const int64_t lda,
                                        int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<double>(handle, m, n, A, lda, ipiv, info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetf2_npvt_64(rocblas_handle handle,
                                        const int64_t m,
                                        const int64_t n,
                                        rocblas_float_complex* A,
                                        const int64_t lda,
                                        int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, info,
                                                                  false);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetf2_npvt_64(rocblas_handle handle,
                                        const int64_t m,
                                        const int64_t n,
                                        rocblas_double_complex* A,
                                        const int64_t lda,
                                        int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getf2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info,
                                                                   false);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

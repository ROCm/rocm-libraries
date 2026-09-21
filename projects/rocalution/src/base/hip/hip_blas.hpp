/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_BLAS_HPP_
#define ROCALUTION_HIP_HIP_BLAS_HPP_

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace rocalution
{
    // rocblas axpy
    template <typename ValueType>
    rocblas_status rocblasTaxpy(rocblas_handle   handle,
                                int              n,
                                const ValueType* alpha,
                                const ValueType* x,
                                int              incx,
                                ValueType*       y,
                                int              incy);

    // rocblas dotu
    template <typename ValueType>
    rocblas_status rocblasTdotu(rocblas_handle   handle,
                                int              n,
                                const ValueType* x,
                                int              incx,
                                const ValueType* y,
                                int              incy,
                                ValueType*       result);

    // rocblas dotconj
    template <typename ValueType>
    rocblas_status rocblasTdotc(rocblas_handle   handle,
                                int              n,
                                const ValueType* x,
                                int              incx,
                                const ValueType* y,
                                int              incy,
                                ValueType*       result);

    // rocprim reduction
    template <typename ValueType>
    hipError_t rocprimTreduce(void*       buffer,
                              size_t&     buffer_size,
                              ValueType*  in,
                              ValueType*  out,
                              size_t      size,
                              hipStream_t stream);

    // rocprim inclusive sum
    template <typename ValueType>
    hipError_t rocprimTinclusivesum(void*       buffer,
                                    size_t&     buffer_size,
                                    ValueType*  in,
                                    ValueType*  out,
                                    size_t      size,
                                    hipStream_t stream);

    // rocprim exclusive sum
    template <typename ValueType>
    hipError_t rocprimTexclusivesum(void*       buffer,
                                    size_t&     buffer_size,
                                    ValueType*  in,
                                    ValueType*  out,
                                    size_t      size,
                                    hipStream_t stream);

    // rocblas nrm2
    template <typename ValueType>
    rocblas_status
        rocblasTnrm2(rocblas_handle handle, int n, const ValueType* x, int incx, ValueType* result);

    // rocblas scal
    template <typename ValueType>
    rocblas_status
        rocblasTscal(rocblas_handle handle, int n, const ValueType* alpha, ValueType* x, int incx);

    // rocblas_amax
    template <typename ValueType>
    rocblas_status
        rocblasTamax(rocblas_handle handle, int n, const ValueType* x, int incx, int* result);

    // rocblas_asum
    template <typename ValueType>
    rocblas_status
        rocblasTasum(rocblas_handle handle, int n, const ValueType* x, int incx, ValueType* result);

    // rocblas_gemv
    template <typename ValueType>
    rocblas_status rocblasTgemv(rocblas_handle    handle,
                                rocblas_operation trans,
                                int               m,
                                int               n,
                                const ValueType*  alpha,
                                const ValueType*  A,
                                int               lda,
                                const ValueType*  x,
                                int               incx,
                                const ValueType*  beta,
                                ValueType*        y,
                                int               incy);

    // rocblas_gemm
    template <typename ValueType>
    rocblas_status rocblasTgemm(rocblas_handle    handle,
                                rocblas_operation transa,
                                rocblas_operation transb,
                                int               m,
                                int               n,
                                int               k,
                                const ValueType*  alpha,
                                const ValueType*  A,
                                int               lda,
                                const ValueType*  B,
                                int               ldb,
                                const ValueType*  beta,
                                ValueType*        C,
                                int               ldc);

    // rocblas_geam
    template <typename ValueType>
    rocblas_status rocblasTgeam(rocblas_handle    handle,
                                rocblas_operation transA,
                                rocblas_operation transB,
                                int               m,
                                int               n,
                                const ValueType*  alpha,
                                const ValueType*  A,
                                int               lda,
                                const ValueType*  beta,
                                const ValueType*  B,
                                int               ldb,
                                ValueType*        C,
                                int               ldc);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_BLAS_HPP_

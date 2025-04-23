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

#include "hip_blas.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocprim/rocprim.hpp>

#include <complex>

namespace rocalution
{
    // rocblas axpy
    template <>
    rocblas_status rocblasTaxpy(rocblas_handle handle,
                                int            n,
                                const float*   alpha,
                                const float*   x,
                                int            incx,
                                float*         y,
                                int            incy)
    {
        return rocblas_saxpy(handle, n, alpha, x, incx, y, incy);
    }

    template <>
    rocblas_status rocblasTaxpy(rocblas_handle handle,
                                int            n,
                                const double*  alpha,
                                const double*  x,
                                int            incx,
                                double*        y,
                                int            incy)
    {
        return rocblas_daxpy(handle, n, alpha, x, incx, y, incy);
    }

    template <>
    rocblas_status rocblasTaxpy(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* alpha,
                                const std::complex<float>* x,
                                int                        incx,
                                std::complex<float>*       y,
                                int                        incy)
    {
        return rocblas_caxpy(handle,
                             n,
                             (const rocblas_float_complex*)alpha,
                             (const rocblas_float_complex*)x,
                             incx,
                             (rocblas_float_complex*)y,
                             incy);
    }

    template <>
    rocblas_status rocblasTaxpy(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* alpha,
                                const std::complex<double>* x,
                                int                         incx,
                                std::complex<double>*       y,
                                int                         incy)
    {
        return rocblas_zaxpy(handle,
                             n,
                             (const rocblas_double_complex*)alpha,
                             (const rocblas_double_complex*)x,
                             incx,
                             (rocblas_double_complex*)y,
                             incy);
    }

    // rocblas dotu
    template <>
    rocblas_status rocblasTdotu(rocblas_handle handle,
                                int            n,
                                const float*   x,
                                int            incx,
                                const float*   y,
                                int            incy,
                                float*         result)
    {
        return rocblas_sdot(handle, n, x, incx, y, incy, result);
    }

    template <>
    rocblas_status rocblasTdotu(rocblas_handle handle,
                                int            n,
                                const double*  x,
                                int            incx,
                                const double*  y,
                                int            incy,
                                double*        result)
    {
        return rocblas_ddot(handle, n, x, incx, y, incy, result);
    }

    template <>
    rocblas_status rocblasTdotu(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* x,
                                int                        incx,
                                const std::complex<float>* y,
                                int                        incy,
                                std::complex<float>*       result)
    {
        return rocblas_cdotu(handle,
                             n,
                             (const rocblas_float_complex*)x,
                             incx,
                             (const rocblas_float_complex*)y,
                             incy,
                             (rocblas_float_complex*)result);
    }

    template <>
    rocblas_status rocblasTdotu(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* x,
                                int                         incx,
                                const std::complex<double>* y,
                                int                         incy,
                                std::complex<double>*       result)
    {
        return rocblas_zdotu(handle,
                             n,
                             (const rocblas_double_complex*)x,
                             incx,
                             (const rocblas_double_complex*)y,
                             incy,
                             (rocblas_double_complex*)result);
    }

    // rocblas dotconj
    template <>
    rocblas_status rocblasTdotc(rocblas_handle handle,
                                int            n,
                                const float*   x,
                                int            incx,
                                const float*   y,
                                int            incy,
                                float*         result)
    {
        return rocblas_sdot(handle, n, x, incx, y, incy, result);
    }

    template <>
    rocblas_status rocblasTdotc(rocblas_handle handle,
                                int            n,
                                const double*  x,
                                int            incx,
                                const double*  y,
                                int            incy,
                                double*        result)
    {
        return rocblas_ddot(handle, n, x, incx, y, incy, result);
    }

    template <>
    rocblas_status rocblasTdotc(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* x,
                                int                        incx,
                                const std::complex<float>* y,
                                int                        incy,
                                std::complex<float>*       result)
    {
        return rocblas_cdotc(handle,
                             n,
                             (const rocblas_float_complex*)x,
                             incx,
                             (const rocblas_float_complex*)y,
                             incy,
                             (rocblas_float_complex*)result);
    }

    template <>
    rocblas_status rocblasTdotc(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* x,
                                int                         incx,
                                const std::complex<double>* y,
                                int                         incy,
                                std::complex<double>*       result)
    {
        return rocblas_zdotc(handle,
                             n,
                             (const rocblas_double_complex*)x,
                             incx,
                             (const rocblas_double_complex*)y,
                             incy,
                             (rocblas_double_complex*)result);
    }

    // rocprim reduce
    template <>
    hipError_t rocprimTreduce(
        void* buffer, size_t& buffer_size, float* in, float* out, size_t size, hipStream_t stream)
    {
        return rocprim::reduce(
            buffer, buffer_size, in, out, 0, size, rocprim::plus<float>(), stream);
    }

    template <>
    hipError_t rocprimTreduce(
        void* buffer, size_t& buffer_size, double* in, double* out, size_t size, hipStream_t stream)
    {
        return rocprim::reduce(
            buffer, buffer_size, in, out, 0, size, rocprim::plus<double>(), stream);
    }

    template <>
    hipError_t rocprimTreduce(void*                buffer,
                              size_t&              buffer_size,
                              std::complex<float>* in,
                              std::complex<float>* out,
                              size_t               size,
                              hipStream_t          stream)
    {
        return rocprim::reduce(buffer,
                               buffer_size,
                               (hipComplex*)in,
                               (hipComplex*)out,
                               0,
                               size,
                               rocprim::plus<hipComplex>(),
                               stream);
    }

    template <>
    hipError_t rocprimTreduce(void*                 buffer,
                              size_t&               buffer_size,
                              std::complex<double>* in,
                              std::complex<double>* out,
                              size_t                size,
                              hipStream_t           stream)
    {
        return rocprim::reduce(buffer,
                               buffer_size,
                               (hipDoubleComplex*)in,
                               (hipDoubleComplex*)out,
                               0,
                               size,
                               rocprim::plus<hipDoubleComplex>(),
                               stream);
    }

    template <>
    hipError_t rocprimTreduce(
        void* buffer, size_t& buffer_size, int* in, int* out, size_t size, hipStream_t stream)
    {
        return rocprim::reduce(buffer, buffer_size, in, out, 0, size, rocprim::plus<int>(), stream);
    }

    template <>
    hipError_t rocprimTreduce(void*       buffer,
                              size_t&     buffer_size,
                              int64_t*    in,
                              int64_t*    out,
                              size_t      size,
                              hipStream_t stream)
    {
        return rocprim::reduce(
            buffer, buffer_size, in, out, 0, size, rocprim::plus<int64_t>(), stream);
    }

    // rocprim inclusive sum
    template <>
    hipError_t rocprimTinclusivesum(
        void* buffer, size_t& buffer_size, float* in, float* out, size_t size, hipStream_t stream)
    {
        return rocprim::inclusive_scan(
            buffer, buffer_size, in, out, size, rocprim::plus<float>(), stream);
    }

    template <>
    hipError_t rocprimTinclusivesum(
        void* buffer, size_t& buffer_size, double* in, double* out, size_t size, hipStream_t stream)
    {
        return rocprim::inclusive_scan(
            buffer, buffer_size, in, out, size, rocprim::plus<double>(), stream);
    }

    template <>
    hipError_t rocprimTinclusivesum(void*                buffer,
                                    size_t&              buffer_size,
                                    std::complex<float>* in,
                                    std::complex<float>* out,
                                    size_t               size,
                                    hipStream_t          stream)
    {
        return rocprim::inclusive_scan(buffer,
                                       buffer_size,
                                       (hipComplex*)in,
                                       (hipComplex*)out,
                                       size,
                                       rocprim::plus<hipComplex>(),
                                       stream);
    }

    template <>
    hipError_t rocprimTinclusivesum(void*                 buffer,
                                    size_t&               buffer_size,
                                    std::complex<double>* in,
                                    std::complex<double>* out,
                                    size_t                size,
                                    hipStream_t           stream)
    {
        return rocprim::inclusive_scan(buffer,
                                       buffer_size,
                                       (hipDoubleComplex*)in,
                                       (hipDoubleComplex*)out,
                                       size,
                                       rocprim::plus<hipDoubleComplex>(),
                                       stream);
    }

    template <>
    hipError_t rocprimTinclusivesum(
        void* buffer, size_t& buffer_size, int* in, int* out, size_t size, hipStream_t stream)
    {
        return rocprim::inclusive_scan(
            buffer, buffer_size, in, out, size, rocprim::plus<int>(), stream);
    }

    template <>
    hipError_t rocprimTinclusivesum(void*       buffer,
                                    size_t&     buffer_size,
                                    int64_t*    in,
                                    int64_t*    out,
                                    size_t      size,
                                    hipStream_t stream)
    {
        return rocprim::inclusive_scan(
            buffer, buffer_size, in, out, size, rocprim::plus<int64_t>(), stream);
    }

    // rocprim exclusive sum
    template <>
    hipError_t rocprimTexclusivesum(
        void* buffer, size_t& buffer_size, float* in, float* out, size_t size, hipStream_t stream)
    {
        return rocprim::exclusive_scan(
            buffer, buffer_size, in, out, 0.0f, size, rocprim::plus<float>(), stream);
    }

    template <>
    hipError_t rocprimTexclusivesum(
        void* buffer, size_t& buffer_size, double* in, double* out, size_t size, hipStream_t stream)
    {
        return rocprim::exclusive_scan(
            buffer, buffer_size, in, out, 0.0, size, rocprim::plus<double>(), stream);
    }

    template <>
    hipError_t rocprimTexclusivesum(void*                buffer,
                                    size_t&              buffer_size,
                                    std::complex<float>* in,
                                    std::complex<float>* out,
                                    size_t               size,
                                    hipStream_t          stream)
    {
        return rocprim::exclusive_scan(buffer,
                                       buffer_size,
                                       (hipComplex*)in,
                                       (hipComplex*)out,
                                       (hipComplex)0,
                                       size,
                                       rocprim::plus<hipComplex>(),
                                       stream);
    }

    template <>
    hipError_t rocprimTexclusivesum(void*                 buffer,
                                    size_t&               buffer_size,
                                    std::complex<double>* in,
                                    std::complex<double>* out,
                                    size_t                size,
                                    hipStream_t           stream)
    {
        return rocprim::exclusive_scan(buffer,
                                       buffer_size,
                                       (hipDoubleComplex*)in,
                                       (hipDoubleComplex*)out,
                                       (hipDoubleComplex)0,
                                       size,
                                       rocprim::plus<hipDoubleComplex>(),
                                       stream);
    }

    template <>
    hipError_t rocprimTexclusivesum(
        void* buffer, size_t& buffer_size, int* in, int* out, size_t size, hipStream_t stream)
    {
        return rocprim::exclusive_scan(
            buffer, buffer_size, in, out, 0, size, rocprim::plus<int>(), stream);
    }

    template <>
    hipError_t rocprimTexclusivesum(void*       buffer,
                                    size_t&     buffer_size,
                                    int64_t*    in,
                                    int64_t*    out,
                                    size_t      size,
                                    hipStream_t stream)
    {
        return rocprim::exclusive_scan(
            buffer, buffer_size, in, out, 0, size, rocprim::plus<int64_t>(), stream);
    }

    // rocblas nrm2
    template <>
    rocblas_status
        rocblasTnrm2(rocblas_handle handle, int n, const float* x, int incx, float* result)
    {
        return rocblas_snrm2(handle, n, x, incx, result);
    }

    template <>
    rocblas_status
        rocblasTnrm2(rocblas_handle handle, int n, const double* x, int incx, double* result)
    {
        return rocblas_dnrm2(handle, n, x, incx, result);
    }

    template <>
    rocblas_status rocblasTnrm2(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* x,
                                int                        incx,
                                std::complex<float>*       result)
    {
        float          res;
        rocblas_status status
            = rocblas_scnrm2(handle, n, (const rocblas_float_complex*)x, incx, &res);

        *result = std::complex<float>(res);
        return status;
    }

    template <>
    rocblas_status rocblasTnrm2(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* x,
                                int                         incx,
                                std::complex<double>*       result)
    {
        double         res;
        rocblas_status status
            = rocblas_dznrm2(handle, n, (const rocblas_double_complex*)x, incx, &res);

        *result = std::complex<double>(res);
        return status;
    }

    // rocblas scal
    template <>
    rocblas_status
        rocblasTscal(rocblas_handle handle, int n, const float* alpha, float* x, int incx)
    {
        return rocblas_sscal(handle, n, alpha, x, incx);
    }

    template <>
    rocblas_status
        rocblasTscal(rocblas_handle handle, int n, const double* alpha, double* x, int incx)
    {
        return rocblas_dscal(handle, n, alpha, x, incx);
    }

    template <>
    rocblas_status rocblasTscal(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* alpha,
                                std::complex<float>*       x,
                                int                        incx)
    {
        return rocblas_cscal(
            handle, n, (const rocblas_float_complex*)alpha, (rocblas_float_complex*)x, incx);
    }

    template <>
    rocblas_status rocblasTscal(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* alpha,
                                std::complex<double>*       x,
                                int                         incx)
    {
        return rocblas_zscal(
            handle, n, (const rocblas_double_complex*)alpha, (rocblas_double_complex*)x, incx);
    }

    // rocblas_amax
    template <>
    rocblas_status rocblasTamax(rocblas_handle handle, int n, const float* x, int incx, int* result)
    {
        return rocblas_isamax(handle, n, x, incx, result);
    }

    template <>
    rocblas_status
        rocblasTamax(rocblas_handle handle, int n, const double* x, int incx, int* result)
    {
        return rocblas_idamax(handle, n, x, incx, result);
    }

    template <>
    rocblas_status rocblasTamax(
        rocblas_handle handle, int n, const std::complex<float>* x, int incx, int* result)
    {
        //    return rocblas_iscamax(handle, n, x, incx, result);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    rocblas_status rocblasTamax(
        rocblas_handle handle, int n, const std::complex<double>* x, int incx, int* result)
    {
        //    return rocblas_idzamax(handle, n, x, incx, result);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // rocblas_asum
    template <>
    rocblas_status
        rocblasTasum(rocblas_handle handle, int n, const float* x, int incx, float* result)
    {
        return rocblas_sasum(handle, n, x, incx, result);
    }

    template <>
    rocblas_status
        rocblasTasum(rocblas_handle handle, int n, const double* x, int incx, double* result)
    {
        return rocblas_dasum(handle, n, x, incx, result);
    }

    template <>
    rocblas_status rocblasTasum(rocblas_handle             handle,
                                int                        n,
                                const std::complex<float>* x,
                                int                        incx,
                                std::complex<float>*       result)
    {
        //    return rocblas_scasum(handle, n, x, incx, result);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    rocblas_status rocblasTasum(rocblas_handle              handle,
                                int                         n,
                                const std::complex<double>* x,
                                int                         incx,
                                std::complex<double>*       result)
    {
        //    return rocblas_dzasum(handle, n, x, incx, result);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // rocblas_gemv
    template <>
    rocblas_status rocblasTgemv(rocblas_handle    handle,
                                rocblas_operation trans,
                                int               m,
                                int               n,
                                const float*      alpha,
                                const float*      A,
                                int               lda,
                                const float*      x,
                                int               incx,
                                const float*      beta,
                                float*            y,
                                int               incy)
    {
        return rocblas_sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template <>
    rocblas_status rocblasTgemv(rocblas_handle    handle,
                                rocblas_operation trans,
                                int               m,
                                int               n,
                                const double*     alpha,
                                const double*     A,
                                int               lda,
                                const double*     x,
                                int               incx,
                                const double*     beta,
                                double*           y,
                                int               incy)
    {
        return rocblas_dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template <>
    rocblas_status rocblasTgemv(rocblas_handle             handle,
                                rocblas_operation          trans,
                                int                        m,
                                int                        n,
                                const std::complex<float>* alpha,
                                const std::complex<float>* A,
                                int                        lda,
                                const std::complex<float>* x,
                                int                        incx,
                                const std::complex<float>* beta,
                                std::complex<float>*       y,
                                int                        incy)
    {
        //    return rocblas_cgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    rocblas_status rocblasTgemv(rocblas_handle              handle,
                                rocblas_operation           trans,
                                int                         m,
                                int                         n,
                                const std::complex<double>* alpha,
                                const std::complex<double>* A,
                                int                         lda,
                                const std::complex<double>* x,
                                int                         incx,
                                const std::complex<double>* beta,
                                std::complex<double>*       y,
                                int                         incy)
    {
        //    return rocblas_zgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // rocblas_gemm
    template <>
    rocblas_status rocblasTgemm(rocblas_handle    handle,
                                rocblas_operation transa,
                                rocblas_operation transb,
                                int               m,
                                int               n,
                                int               k,
                                const float*      alpha,
                                const float*      A,
                                int               lda,
                                const float*      B,
                                int               ldb,
                                const float*      beta,
                                float*            C,
                                int               ldc)
    {
        return rocblas_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <>
    rocblas_status rocblasTgemm(rocblas_handle    handle,
                                rocblas_operation transa,
                                rocblas_operation transb,
                                int               m,
                                int               n,
                                int               k,
                                const double*     alpha,
                                const double*     A,
                                int               lda,
                                const double*     B,
                                int               ldb,
                                const double*     beta,
                                double*           C,
                                int               ldc)
    {
        return rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <>
    rocblas_status rocblasTgemm(rocblas_handle             handle,
                                rocblas_operation          transa,
                                rocblas_operation          transb,
                                int                        m,
                                int                        n,
                                int                        k,
                                const std::complex<float>* alpha,
                                const std::complex<float>* A,
                                int                        lda,
                                const std::complex<float>* B,
                                int                        ldb,
                                const std::complex<float>* beta,
                                std::complex<float>*       C,
                                int                        ldc)
    {
        //    return rocblas_cgemm(handle, transa, transb, m, n, k,
        //                         alpha, A, lda, B, ldb, beta, C, ldc);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    rocblas_status rocblasTgemm(rocblas_handle              handle,
                                rocblas_operation           transa,
                                rocblas_operation           transb,
                                int                         m,
                                int                         n,
                                int                         k,
                                const std::complex<double>* alpha,
                                const std::complex<double>* A,
                                int                         lda,
                                const std::complex<double>* B,
                                int                         ldb,
                                const std::complex<double>* beta,
                                std::complex<double>*       C,
                                int                         ldc)
    {
        //    return rocblas_zgemm(handle, transa, transb, m, n, k,
        //                         alpha, A, lda, B, ldb, beta, C, ldc);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // rocblas_geam
    template <>
    rocblas_status rocblasTgeam(rocblas_handle    handle,
                                rocblas_operation transA,
                                rocblas_operation transB,
                                int               m,
                                int               n,
                                const float*      alpha,
                                const float*      A,
                                int               lda,
                                const float*      beta,
                                const float*      B,
                                int               ldb,
                                float*            C,
                                int               ldc)
    {
        return rocblas_sgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    template <>
    rocblas_status rocblasTgeam(rocblas_handle    handle,
                                rocblas_operation transA,
                                rocblas_operation transB,
                                int               m,
                                int               n,
                                const double*     alpha,
                                const double*     A,
                                int               lda,
                                const double*     beta,
                                const double*     B,
                                int               ldb,
                                double*           C,
                                int               ldc)
    {
        return rocblas_dgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    template <>
    rocblas_status rocblasTgeam(rocblas_handle             handle,
                                rocblas_operation          transA,
                                rocblas_operation          transB,
                                int                        m,
                                int                        n,
                                const std::complex<float>* alpha,
                                const std::complex<float>* A,
                                int                        lda,
                                const std::complex<float>* beta,
                                const std::complex<float>* B,
                                int                        ldb,
                                std::complex<float>*       C,
                                int                        ldc)
    {
        return rocblas_cgeam(handle,
                             transA,
                             transB,
                             m,
                             n,
                             (const rocblas_float_complex*)alpha,
                             (const rocblas_float_complex*)A,
                             lda,
                             (const rocblas_float_complex*)beta,
                             (const rocblas_float_complex*)B,
                             ldb,
                             (rocblas_float_complex*)C,
                             ldc);
    }

    template <>
    rocblas_status rocblasTgeam(rocblas_handle              handle,
                                rocblas_operation           transA,
                                rocblas_operation           transB,
                                int                         m,
                                int                         n,
                                const std::complex<double>* alpha,
                                const std::complex<double>* A,
                                int                         lda,
                                const std::complex<double>* beta,
                                const std::complex<double>* B,
                                int                         ldb,
                                std::complex<double>*       C,
                                int                         ldc)
    {
        return rocblas_zgeam(handle,
                             transA,
                             transB,
                             m,
                             n,
                             (const rocblas_double_complex*)alpha,
                             (const rocblas_double_complex*)A,
                             lda,
                             (const rocblas_double_complex*)beta,
                             (const rocblas_double_complex*)B,
                             ldb,
                             (rocblas_double_complex*)C,
                             ldc);
    }

} // namespace rocalution

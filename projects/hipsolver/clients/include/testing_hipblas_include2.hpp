/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

// test with hipBLAS included after hipSOLVER
#include "testing_ormtr_unmtr.hpp"

#include "hipblas/hipblas.h"

template <typename T, bool COMPLEX = is_complex<T>>
void testing_hipblas_include2(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   sideC  = argus.get<char>("side");
    char                   uploC  = argus.get<char>("uplo");
    char                   transC = argus.get<char>("trans");
    int                    m, n;
    if(sideC == 'L')
    {
        m = argus.get<int>("m");
        n = argus.get<int>("n", m);
    }
    else
    {
        n = argus.get<int>("n");
        m = argus.get<int>("m", n);
    }
    int nq  = (sideC == 'L' ? m : n);
    int lda = argus.get<int>("lda", nq);
    int ldc = argus.get<int>("ldc", m);

    hipsolverSideMode_t  side      = char2hipsolver_side(sideC);
    hipsolverFillMode_t  uplo      = char2hipsolver_fill(uploC);
    hipsolverOperation_t trans     = char2hipsolver_operation(transC);
    int                  hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value
        = ((COMPLEX && trans == HIPSOLVER_OP_T) || (!COMPLEX && trans == HIPSOLVER_OP_C));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(false,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    trans,
                                                    m,
                                                    n,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);
        return;
    }

    // determine sizes
    bool   left   = (side == HIPSOLVER_SIDE_LEFT);
    size_t size_P = size_t(nq);
    size_t size_C = size_t(ldc) * n;

    size_t size_A    = size_t(lda) * nq;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || ldc < m || lda < nq);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(false,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    trans,
                                                    m,
                                                    n,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);
        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_ormtr_unmtr_bufferSize(false,
                                     handle,
                                     side,
                                     uplo,
                                     trans,
                                     m,
                                     n,
                                     (T*)nullptr,
                                     lda,
                                     (T*)nullptr,
                                     (T*)nullptr,
                                     ldc,
                                     &size_W);

    // memory allocations
    host_strided_batch_vector<T>     hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T>     hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<T>     hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    host_strided_batch_vector<int>   hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T>   dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    ormtr_unmtr_getError<false, T>(handle,
                                   side,
                                   uplo,
                                   trans,
                                   m,
                                   n,
                                   dA,
                                   lda,
                                   dIpiv,
                                   dC,
                                   ldc,
                                   dWork,
                                   size_W,
                                   dInfo,
                                   hA,
                                   hIpiv,
                                   hC,
                                   hCRes,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    int s = left ? m : n;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // ensure all arguments were consumed
    argus.validate_consumed();
}

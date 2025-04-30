/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include <assert.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define ROCM_MATHLIBS_API_USE_HIP_COMPLEX
#include <rocblas/rocblas.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

int main(int argc, char** argv)
{
    int N = 267;
    if(argc > 1)
        N = atoi(argv[1]);

    size_t lda;
    size_t rows, cols;
    int    incx, incy;

    rows = N;
    cols = N;
    lda  = N;
    incx = incy = 1;

    size_t sizeA = size_t(cols) * lda;
    size_t sizeX = size_t(N) * incx;
    size_t sizeY = size_t(N) * incy;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    std::vector<hipFloatComplex> hA(sizeA);
    std::vector<hipFloatComplex> hResult(sizeA);
    std::vector<hipFloatComplex> hX(sizeX);
    std::vector<hipFloatComplex> hY(sizeY);

    for(int i1 = 0; i1 < N; i1++)
    {
        hX[i1 * incx] = hipFloatComplex(1.0f, 0.0f);
        hY[i1 * incy] = hipFloatComplex(1.0f, 0.0f);
    }

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            hA[i1 + i2 * lda] = hipFloatComplex((float)(rand() % 10), 0.0f);

    hipFloatComplex* dA = nullptr;
    hipFloatComplex* dX = nullptr;
    hipFloatComplex* dY = nullptr;
    CHECK_HIP_ERROR(hipMalloc((void**)&dA, sizeof(hipFloatComplex) * sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&dX, sizeof(hipFloatComplex) * sizeX));
    CHECK_HIP_ERROR(hipMalloc((void**)&dY, sizeof(hipFloatComplex) * sizeY));

    // scalar arguments will be from host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    const rocblas_fill uplo   = rocblas_fill_upper;
    hipFloatComplex    hAlpha = hipFloatComplex(2.0f, 0.0f);

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dX, hX.data(), sizeof(hipFloatComplex) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dY, hY.data(), sizeof(hipFloatComplex) * sizeY, hipMemcpyHostToDevice));

    rstatus = rocblas_set_matrix(rows, cols, sizeof(hipFloatComplex), hA.data(), lda, dA, lda);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    // API is defined as using hipFloatComplex types so no casting required
    rstatus = rocblas_cher2(handle, uplo, N, &hAlpha, dX, incx, dY, incy, dA, lda);
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch results
    rstatus = rocblas_get_matrix(rows, cols, sizeof(hipFloatComplex), dA, lda, hResult.data(), lda);
    CHECK_ROCBLAS_STATUS(rstatus);

    // check against expected results for upper and numeric inputs
    bool fail = false;
    for(size_t i1 = 0; i1 < rows; i1++)
        for(size_t i2 = 0; i2 < cols; i2++)
            if(i1 <= i2
               && hResult[i1 + i2 * lda]
                      != hipCaddf(hA[i1 + i2 * lda], hipFloatComplex(4.0f * hX[i1 * incx].x, 0.0f)))
                fail = true;

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dX));
    CHECK_HIP_ERROR(hipFree(dY));

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    fprintf(stdout, "%s\n", fail ? "FAIL" : "PASS");

    return 0;
}

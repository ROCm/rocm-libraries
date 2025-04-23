/*
Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <assert.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

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
    size_t lda, ldb, lddev;
    size_t rows, cols;

    int n = 267;
    if(argc > 1)
        n = atoi(argv[1]);

    rows = n;
    cols = 2 * n;
    lda = ldb = lddev = n;

    typedef double data_type;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    hipStream_t test_stream;
    rstatus = rocblas_get_stream(handle, &test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type* ha;
    data_type* hb;

    // allocate pinned memory to allow async memory transfer
    CHECK_HIP_ERROR(
        hipHostMalloc((void**)&ha, lda * cols * sizeof(data_type), hipHostMallocMapped));
    CHECK_HIP_ERROR(
        hipHostMalloc((void**)&hb, ldb * cols * sizeof(data_type), hipHostMallocMapped));

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            ha[i1 + i2 * lda] = 1.0;

    data_type* da = 0;
    data_type* db = 0;
    data_type* dc = 0;
    CHECK_HIP_ERROR(hipMalloc((void**)&da, lddev * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void**)&db, lddev * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void**)&dc, lddev * cols * sizeof(data_type)));

    // upload asynchronously from pinned memory
    rstatus
        = rocblas_set_matrix_async(rows, cols, sizeof(data_type), ha, lda, da, lddev, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);
    rstatus
        = rocblas_set_matrix_async(rows, cols, sizeof(data_type), ha, lda, db, lddev, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    // scalar arguments will be from host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type alpha = 1.0;
    data_type beta  = 2.0;

    // invoke asynchronous computation
    rstatus = rocblas_dgeam(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            rows,
                            cols,
                            &alpha,
                            da,
                            lddev,
                            &beta,
                            db,
                            lddev,
                            dc,
                            lddev);
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch results asynchronously to pinned memory
    rstatus
        = rocblas_get_matrix_async(rows, cols, sizeof(data_type), dc, lddev, hb, ldb, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    // wait on transfer to be finished
    CHECK_HIP_ERROR(hipStreamSynchronize(test_stream));

    // check against expected results
    bool fail = false;
    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            if(hb[i1 + i2 * ldb] != (alpha + beta) * ha[i1 + i2 * lda])
                fail = true;

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));

    // free pinned memory
    CHECK_HIP_ERROR(hipHostFree(ha));
    CHECK_HIP_ERROR(hipHostFree(hb));

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    fprintf(stdout, "%s\n", fail ? "FAIL" : "PASS");

    return 0;
}

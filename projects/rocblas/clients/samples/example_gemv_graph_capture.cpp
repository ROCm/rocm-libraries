/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "client_utility.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

const int NUM_ITER = 1000;

double maxRelativeError(std::vector<double>& A, std::vector<double>& reference)
{
    double maxRelativeError = double(std::numeric_limits<double>::min());

    size_t n = A.size();
    for(size_t i = 0; i < n; ++i)
    {
        double gold          = double(reference[i]);
        double relativeError = gold != 0 ? (gold - double(A[i])) / (gold) : double(A[i]);
        relativeError        = relativeError > 0 ? relativeError : -relativeError;
        maxRelativeError     = relativeError < maxRelativeError ? maxRelativeError : relativeError;
    }
    return maxRelativeError;
}

int main()
{
    rocblas_status rstatus = rocblas_status_success;
    rocblas_int    M       = 8192;
    rocblas_int    N       = 3;
    rocblas_int    incx    = 1;
    rocblas_int    incy    = 1;
    typedef double dataType;

    dataType h_alpha = 1.0;
    dataType h_beta  = 0.0;

    const rocblas_operation trans_A = rocblas_operation_transpose;

    size_t size_x, dim_x, abs_incx;
    size_t size_y, dim_y, abs_incy;

    if(trans_A == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else // transpose
    {
        dim_x = M;
        dim_y = N;
    }

    rocblas_int lda    = M;
    size_t      size_A = lda * size_t(dim_x);

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;
    size_x   = dim_x * abs_incx;
    size_y   = dim_y * abs_incy;

    std::vector<dataType> hA(size_A);
    std::vector<dataType> hx(size_x);
    std::vector<dataType> hy(size_y, 0);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_A; ++i)
    {
        hA[i] = rand() % 17;
    }
    for(int i = 0; i < size_x; ++i)
    {
        hx[i] = rand() % 17;
    }

    std::vector<dataType> hy_gold = hy;

    // using rocblas API
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    hipStream_t stream_1, stream_2;
    CHECK_HIP_ERROR(hipStreamCreate(&stream_1));
    CHECK_HIP_ERROR(hipStreamCreate(&stream_2));

    bool           graph_captured = false;
    hipGraph_t     graph;
    hipGraphExec_t graph_exec;

    // allocate memory on device
    dataType *dA, *dX, *dY;
    CHECK_HIP_ERROR(hipMalloc(&dA, size_A * sizeof(dataType)));
    CHECK_HIP_ERROR(hipMalloc(&dX, size_x * sizeof(dataType)));
    CHECK_HIP_ERROR(hipMalloc(&dY, size_y * sizeof(dataType)));

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(dataType) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dX, hx.data(), sizeof(dataType) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dY, hy.data(), sizeof(dataType) * size_y, hipMemcpyHostToDevice));

    // enable passing alpha and beta parameters from pointer to host memory
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, stream_1));

    for(auto i = 0; i < NUM_ITER; i++)
    {
        if(!graph_captured)
        {
            CHECK_HIP_ERROR(hipStreamBeginCapture(stream_1, hipStreamCaptureModeGlobal));
            // asynchronous calculation on device, returns before finished calculations
            CHECK_ROCBLAS_ERROR(rocblas_dgemv(
                handle, trans_A, M, N, &h_alpha, dA, lda, dX, incx, &h_beta, dY, incy));
            CHECK_HIP_ERROR(hipStreamEndCapture(stream_1, &graph));
            CHECK_HIP_ERROR(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
            CHECK_HIP_ERROR(hipGraphDestroy(graph));
            graph_captured = true;
        }
        CHECK_HIP_ERROR(
            hipGraphLaunch(graph_exec, stream_2)); //Launching the graph in a different stream.
    }

    // fetch device memory results, automatically blocked until results ready
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dY, sizeof(dataType) * size_y, hipMemcpyDeviceToHost));

    rocblas_cout << "M, N, lda = " << M << ", " << N << ", " << lda << std::endl;

    // calculate expected result using CPU
    for(size_t j = 0; j < size_y; j++)
    {
        for(size_t i = j * M, k = 0; k < M; k++, i++)
        {
            hy_gold[j] += h_alpha * hA[i] * hx[k] + h_beta * hy_gold[j];
        }
    }

    auto     maxRelativeErr = maxRelativeError(hy, hy_gold);
    dataType eps            = std::numeric_limits<dataType>::epsilon();
    dataType tolerance      = 10;
    if(maxRelativeErr > eps * tolerance)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }
    std::cout << ": max. relative err. = " << maxRelativeErr << std::endl;

    CHECK_HIP_ERROR(hipGraphExecDestroy(graph_exec));

    /* CAUTION: Do not destroy streams before destroying the rocblas_handle.
       rocblas_handle requires stream parameter to free allocated device memory.*/
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    CHECK_HIP_ERROR(hipStreamDestroy(stream_1));
    CHECK_HIP_ERROR(hipStreamDestroy(stream_2));

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dX));
    CHECK_HIP_ERROR(hipFree(dY));

    return EXIT_SUCCESS;
}

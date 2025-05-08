/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm> // for std::min
#include <hip/hip_runtime_api.h> // for hip functions
#include <hipsolver/hipsolver.h> // for all the hipsolver C interfaces and type declarations
#include <stdio.h> // for size_t, printf
#include <vector>

// Example: Compute the LU Factorization of a matrix on the GPU

void get_example_matrix(std::vector<double>& hA, int& M, int& N, int& lda)
{
    // a *very* small example input; not a very efficient use of the API
    const double A[3][3] = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
    M                    = 3;
    N                    = 3;
    lda                  = 3;
    // note: matrices must be stored in column major format,
    //       i.e. entry (i,j) should be accessed by hA[i + j*lda]
    hA.resize(size_t(lda) * N);
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            // copy A (2D array) into hA (1D array, column-major)
            hA[i + j * lda] = A[i][j];
        }
    }
}

// We use hipsolverDgetrf to factor a real M-by-N matrix, A.
int main()
{
    int                 M; // rows
    int                 N; // cols
    int                 lda; // leading dimension
    std::vector<double> hA; // input matrix on CPU
    get_example_matrix(hA, M, N, lda);

    // let's print the input matrix, just to see it
    printf("A = [\n");
    for(size_t i = 0; i < M; ++i)
    {
        printf("  ");
        for(size_t j = 0; j < N; ++j)
        {
            printf("% .3f ", hA[i + j * lda]);
        }
        printf(";\n");
    }
    printf("]\n");

    // initialization
    hipsolverHandle_t handle;
    hipsolverCreate(&handle);

    // calculate the sizes of our arrays
    size_t size_piv = size_t(std::min(M, N)); // count of pivot indices
    size_t size_A   = size_t(lda) * N; // count of elements in matrix A

    // allocate memory on GPU
    int*    dInfo;
    int*    dIpiv;
    double* dA;
    hipMalloc(&dInfo, sizeof(int));
    hipMalloc(&dIpiv, sizeof(int) * size_piv);
    hipMalloc(&dA, sizeof(double) * size_A);

    // copy data to GPU
    hipMemcpy(dA, hA.data(), sizeof(double) * size_A, hipMemcpyHostToDevice);

    // create the workspace
    double* dWork;
    int     size_work; // size of workspace to pass to getrf
    hipsolverDgetrf_bufferSize(handle, M, N, dA, lda, &size_work);
    hipMalloc(&dWork, size_work);

    // compute the LU factorization on the GPU
    hipsolverStatus_t status
        = hipsolverDgetrf(handle, M, N, dA, lda, dWork, size_work, dIpiv, dInfo);
    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    // copy the results back to CPU
    std::vector<int> hInfo(1); // provides information about algorithm completion
    std::vector<int> hIpiv(size_piv); // array for pivot indices on CPU
    hipMemcpy(hInfo.data(), dInfo, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(hIpiv.data(), dIpiv, sizeof(int) * size_piv, hipMemcpyDeviceToHost);
    hipMemcpy(hA.data(), dA, sizeof(double) * size_A, hipMemcpyDeviceToHost);

    // the results are now in hA and hIpiv
    // we can print some of the results if we want to see them
    printf("U = [\n");
    for(size_t i = 0; i < M; ++i)
    {
        printf("  ");
        for(size_t j = 0; j < N; ++j)
        {
            printf("% .3f ", (i <= j) ? hA[i + j * lda] : 0);
        }
        printf(";\n");
    }
    printf("]\n");

    // clean up
    hipFree(dWork);
    hipFree(dInfo);
    hipFree(dIpiv);
    hipFree(dA);
    hipsolverDestroy(handle);
}

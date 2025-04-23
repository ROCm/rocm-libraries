/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::float64_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

// Host dgemv validation
__host__ void dgemv_cpu_h(uint32_t         m,
                          uint32_t         n,
                          uint32_t         k,
                          float64_t const* a,
                          float64_t const* b,
                          float64_t*       c,
                          float64_t        alpha,
                          float64_t        beta)
{
    uint32_t lda = m;

    for(int i = 0; i < m; ++i)
    {
        float64_t accum = 0.0f;
        for(int h = 0; h < k; ++h)
        {
            accum += a[h * lda + i] * b[h];
        }
        c[i] = alpha * accum + beta * c[i];
    }
}

// Supports ROCWMMA_M/N square sizes of
// : 16 x 16
// : 32 x 32 ( only MI )
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;

// Supports ROCWMMA_K sizes as
// : multiples of 16.
const int ROCWMMA_K = 16;

// AMDGCN default wave size
const uint32_t WAVE_SIZE = getWarpSize();

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one ROCWMMA_M x ROCWMMA_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X = 16 * WAVE_SIZE;
const int T_BLOCK_Y = 1;

// The following device kernel is a naive implementation
// of blocked dgemv. Each wave will compute one ROCWMMA_M x ROCWMMA_N
// output block of the m x k x 1 dgemv, generalized as:
// y = alpha * (A) * x + beta * y
//
// In this simplified example, we assume:
//  A - Matrix of size m * k (col-major)
//  x - Vector of size k * 1 (col-major)
//  y - accumulator of size m * 1 (col-major)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level dgemv computation, and is not optimized.
__global__ void dgemv_rocwmma_d(uint32_t         m,
                                uint32_t         n,
                                uint32_t         k,
                                float64_t const* a,
                                float64_t const* b,
                                float64_t*       c,
                                float64_t*       d,
                                uint32_t         lda,
                                uint32_t         ldb,
                                uint32_t         ldc,
                                uint32_t         ldd,
                                float64_t        alpha,
                                float64_t        beta)
{
    // Create frags
    auto fragA
        = rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major>();
    auto fragB
        = rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major>();
    auto fragC   = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t>();

    rocwmma::fill_fragment(fragAcc, 0.0);

    int majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;

    // Target C block
    int cRow = majorWarp * ROCWMMA_M;

    if(cRow < m)
    {
        // fragAcc = A x B
        for(int i = 0; i < k; i += ROCWMMA_K)
        {
            // Load the inputs
            rocwmma::load_matrix_sync(fragA, a + (cRow + i * lda), lda);
            rocwmma::load_matrix_sync(fragB, b + i, ldb);

            // Matrix multiply - accumulate using MFMA units
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        rocwmma::load_matrix_sync(fragC, c + cRow, ldc, rocwmma::mem_col_major);

        for(int i = 0; i < fragC.num_elements; i++)
        {
            fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
        }

        // Store the output
        rocwmma::store_matrix_sync(d + cRow, fragC, ldc, rocwmma::mem_col_major);
    }
}

__host__ void dgemv_test(uint32_t m, uint32_t n, uint32_t k, float64_t alpha, float64_t beta)
{
    // Bounds check
    if(m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K)
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    int lda = m;
    int ldb = k;
    int ldc = m;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;
    // Initialize input matrices
    std::vector<float64_t> matrixA(m * k); // matrix
    std::vector<float64_t> matrixB(k * 1); // vector
    std::vector<float64_t> matrixC(m * 1, 1.0); //accum

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, 1);

    std::cout << "Initializing device data..." << std::endl;
    // Allocate and copy device memory
    float64_t* d_a;
    float64_t* d_b;
    float64_t* d_c;

    const size_t bytesA = matrixA.size() * sizeof(float64_t);
    const size_t bytesB = matrixB.size() * sizeof(float64_t);
    const size_t bytesC = matrixC.size() * sizeof(float64_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));

    std::cout << "Launching dgemv kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(dgemv_rocwmma_d,
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_c,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, " << std::endl;

    std::cout << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << std::endl;

#if !NDEBUG

    std::cout << "Validating result with reference..." << std::endl;
    // Bring kernel result back to host
    std::vector<float64_t> matrixC_device(m * 1, std::numeric_limits<float64_t>::signaling_NaN());
    CHECK_HIP_ERROR(hipMemcpy(matrixC_device.data(), d_c, bytesC, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float64_t> matrixC_host(matrixC);
    dgemv_cpu_h(m, n, k, matrixA.data(), matrixB.data(), matrixC_host.data(), alpha, beta);

    auto res = compareEqual<float64_t>(matrixC_host.data(), matrixC_device.data(), m);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED!\n";
    }
    else
    {
        std::cout << "PASSED!\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    const uint32_t m = 256;
    const uint32_t k = 256;
    const uint32_t n = T_BLOCK_Y * ROCWMMA_N;

    if(!isF64Supported())
    {
        std::cout << "f64 dgemv not supported on this device" << std::endl;
    }
    else
    {
        dgemv_test(m, n, k, 2.1, 2.1);
    }

    return 0;
}

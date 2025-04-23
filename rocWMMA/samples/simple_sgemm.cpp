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
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

// Supports ROCWMMA_M/N square sizes of
// : 16 x 16
// : 32 x 32 ( only MI )
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;

// Supports ROCWMMA_K sizes as
// : multiples of 16.
const int ROCWMMA_K = 16;

// Device warp size
const uint32_t WAVE_SIZE = getWarpSize();

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

// The following device kernel is a naive implementation
// of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the M x N x K GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this simplified example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level GEMM computation, and is not optimized.
__global__ void sgemm_rocwmma_d(uint32_t         m,
                                uint32_t         n,
                                uint32_t         k,
                                float32_t const* a,
                                float32_t const* b,
                                float32_t const* c,
                                float32_t*       d,
                                uint32_t         lda,
                                uint32_t         ldb,
                                uint32_t         ldc,
                                uint32_t         ldd,
                                float32_t        alpha,
                                float32_t        beta)
{
    // Create frags
    auto fragA
        = rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t, row_major>();
    auto fragB
        = rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t, col_major>();
    auto fragC   = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    // Tile using a 2D grid
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    // Target C block
    auto cRow = majorWarp * ROCWMMA_M;
    auto cCol = minorWarp * ROCWMMA_N;

    // Bounds check
    if(cRow < m && cCol < n)
    {
        // fragAcc = A x B
        for(int i = 0; i < k; i += ROCWMMA_K)
        {
            // Load the inputs
            rocwmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
            rocwmma::load_matrix_sync(fragB, b + (i + cCol * ldb), ldb);

            // Matrix multiply - accumulate using MFMA units
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Fetch C matrix
        rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

        // D = alpha * A x B + beta * C
        for(int i = 0; i < fragC.num_elements; ++i)
        {
            fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
        }

        // Store to D
        rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);
    }
}

__host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float32_t alpha, float32_t beta)
{
    // Bounds check
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
       || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    int lda = k;
    int ldb = k;
    int ldc = n;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<float32_t> matrixA(m * k);
    std::vector<float32_t> matrixB(k * n);
    std::vector<float32_t> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    float32_t* d_a;
    float32_t* d_b;
    float32_t* d_c;
    float32_t* d_d;

    const size_t bytesA = matrixA.size() * sizeof(float32_t);
    const size_t bytesB = matrixB.size() * sizeof(float32_t);
    const size_t bytesC = matrixC.size() * sizeof(float32_t);
    const size_t bytesD = matrixD.size() * sizeof(float32_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));

    std::cout << "Launching GEMM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(sgemm_rocwmma_d,
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
                          d_d,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2*mnk
    auto gFlops       = calculateGFlops(m, n, k);
    auto tFlopsPerSec = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs));

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

#if !NDEBUG

    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float32_t> matrixD_ref(m * n, std::numeric_limits<float32_t>::signaling_NaN());
    gemm_cpu_h<float32_t, float32_t, float32_t, row_major, col_major, row_major>(m,
                                                                                 n,
                                                                                 k,
                                                                                 matrixA.data(),
                                                                                 matrixB.data(),
                                                                                 matrixC.data(),
                                                                                 matrixD_ref.data(),
                                                                                 lda,
                                                                                 ldb,
                                                                                 ldc,
                                                                                 ldd,
                                                                                 alpha,
                                                                                 beta);

    auto res = compareEqual<float32_t>(matrixD.data(), matrixD_ref.data(), m * n);

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
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    if(!isF32Supported())
    {
        std::cout << "f32 sgemm not supported on this device" << std::endl;
    }
    else
    {
        gemm_test(256, 256, 256, 2.1f, 2.1f);
    }
    return 0;
}

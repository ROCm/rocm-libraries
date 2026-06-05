/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP RTC (Runtime Compilation) regression tests with the rocWMMA headers
// These tests ensure that the HIP RTC flow works when we include rocwmma.hpp,
// and that we are able to compile and run a simple GEMM kernel written using
// rocWMMA.

#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <string>
#include <vector>

#include <rocwmma/rocwmma.hpp>

#include "regression/reg_common.hpp"
#include "reference.hpp"

static std::string binary_name;

// ============================================================================
// Kernel Sources
// ============================================================================

// Simple vector addition kernel with rocWMMA include
static const char* vectorAddSourceRocwmmaInclude = R"(

// FIXME: Remove once ROCM-864 is resolved.
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
// FIXME

#include <rocwmma/rocwmma.hpp>
extern "C"
__global__ void vectorAdd(const int* a, const int* b, int* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
)";

static const char* basicRocwmmaGemmSource = R"(

// FIXME: Remove once ROCM-864 is resolved.
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
// FIXME

#include <rocwmma/rocwmma.hpp>

using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;
using rocwmma::bfloat16_t;
using rocwmma::uint32_t;


using InputT   = bfloat16_t;
using OutputT  = float32_t;
using ComputeT = float32_t;

constexpr int ROCWMMA_M = 16;
constexpr int ROCWMMA_N = 16;
constexpr int ROCWMMA_K = 16;

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
extern "C"
__global__ void gemm_rocwmma_d(uint32_t         m,
                               uint32_t         n,
                               uint32_t         k,
                               InputT const* a,
                               InputT const* b,
                               OutputT const* c,
                               OutputT*       d,
                               uint32_t         lda,
                               uint32_t         ldb,
                               uint32_t         ldc,
                               uint32_t         ldd,
                               ComputeT        alpha,
                               ComputeT        beta)
{
    // Create frags
    auto fragA = rocwmma::fragment<rocwmma::matrix_a,
                                   ROCWMMA_M,
                                   ROCWMMA_N,
                                   ROCWMMA_K,
                                   InputT,
                                   rocwmma::row_major>();
    auto fragB = rocwmma::fragment<rocwmma::matrix_b,
                                   ROCWMMA_M,
                                   ROCWMMA_N,
                                   ROCWMMA_K,
                                   InputT,
                                   rocwmma::col_major>();
    auto fragC
        = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT>();
    auto fragAcc
        = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>();

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
)";


// ============================================================================
// Test Suite
// ============================================================================

class HipRTC_rocWMMA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if a ROCm device is available
        int deviceCount = 0;
        CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

        if(deviceCount == 0)
        {
            GTEST_SKIP() << "No ROCm-capable device available";
        }
    }
};


// ============================================================================
// Test 1: Basic rocWMMA include chain
// ============================================================================
// Tests that HIP RTC works with the rocWMMA headers
// - Simple kernel that includes the rocWMMA header

TEST_F(HipRTC_rocWMMA, RocwmmaBasicIncludeTest)
{
    // Test parameters
    const int n         = 1024;
    const int blockSize = 256;
    const int gridSize  = rocwmma::ceil_div(n, blockSize);

    // Initialize host data
    std::vector<int> h_a(n);
    std::vector<int> h_b(n);
    std::vector<int> h_c(n);
    std::vector<int> h_reference(n);

    // Fill with sequential values for predictable debugging
    for(int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Compute reference result on CPU
    vectorAddCPU(h_a.data(), h_b.data(), h_reference.data(), n);

    // Allocate device memory
    int* d_a;
    int* d_b;
    int* d_c;
    CHECK_HIP_ERROR(hipMalloc(&d_a, n * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, n * sizeof(int)));

    // Copy input data to device
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), n * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Create HIP RTC program
    hiprtcProgram prog;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog, vectorAddSourceRocwmmaInclude, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto        includeDirArg = getIncludeDirArg(binary_name);
    auto        options       = buildCompileOptions();
    if(!includeDirArg.empty())
        options.push_back(includeDirArg.c_str());
    int  numOptions = options.size();

    // Compile the program
    hiprtcResult compileResult = hiprtcCompileProgram(prog, numOptions, options.data());

    // Handle compilation errors
    if(compileResult != HIPRTC_SUCCESS)
    {
        std::size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::string options_str = "";
        for (auto s: options) {
            options_str += std::string(s) + " ";
        }
        FAIL() << "Build Options: " << options_str << "\nCompilation log:\n" << log;
    }

    // Extract compiled code
    std::size_t code_size;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCode(prog, code.data()));

    // Load module and get function
    hipModule_t   module;
    hipFunction_t func;
    CHECK_HIP_ERROR(hipModuleLoadData(&module, code.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func, module, "vectorAdd"));

    // Set up kernel parameters using parameter buffer
    struct
    {
        int* d_a;
        int* d_b;
        int* d_c;
        int  n;
    } args{d_a, d_b, d_c, n};

    std::size_t args_size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size,
                      HIP_LAUNCH_PARAM_END};

    // Launch kernel
    CHECK_HIP_ERROR(hipModuleLaunchKernel(func,
                                          gridSize,
                                          1,
                                          1,
                                          blockSize,
                                          1,
                                          1,
                                          0,
                                          nullptr,
                                          nullptr,
                                          (void**)&config));

    // Wait for completion
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_c.data(), d_c, n * sizeof(int), hipMemcpyDeviceToHost));

    // Verify results
    for(int i = 0; i < n; i++)
    {
        EXPECT_EQ(h_c[i], h_reference[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    CHECK_HIP_ERROR(hipModuleUnload(module));
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
}

// ============================================================================
// Test 2: Basic rocWMMA GEMM
// ============================================================================
// Tests that HIP RTC works with the rocWMMA headers
// - Simple GEMM kernel using the rocWMMA header

TEST_F(HipRTC_rocWMMA, RocwmmaGemmTest)
{
    using rocwmma::bfloat16_t;
    using rocwmma::float16_t;
    using rocwmma::float32_t;
    using rocwmma::float64_t;

    using rocwmma::col_major;
    using rocwmma::row_major;

    // Types
    using InputT   = bfloat16_t;
    using OutputT  = float32_t;
    using ComputeT = float32_t;

    // Supports ROCWMMA_M/N square sizes of
    // : 16 x 16
    // : 32 x 32
    const int ROCWMMA_M = 16;
    const int ROCWMMA_N = 16;

    // Supports ROCWMMA_K sizes as
    // : multiples of 16.
    const int ROCWMMA_K = 16;

    // Device warp size
    const int WAVE_SIZE = getWarpSize();

    // Thread block
    // : T_BLOCK_X must be multiple of WAVE_SIZE.
    // Note: Each wave will compute one BLOCK_M x BLOCK_N output block
    // Note: Workgroup will compute
    //  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
    const int T_BLOCK_X = 4 * WAVE_SIZE;
    const int T_BLOCK_Y = 4;

    // gemm parameters
    uint32_t m     = 256;
    uint32_t n     = 256;
    uint32_t k     = 256;
    ComputeT alpha = 2.1f;
    ComputeT beta  = 2.1f;

    // Create HIP RTC program
    hiprtcProgram prog;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog, basicRocwmmaGemmSource, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto        includeDirArg = getIncludeDirArg(binary_name);
    auto        options       = buildCompileOptions();
    if(!includeDirArg.empty())
        options.push_back(includeDirArg.c_str());
    int  numOptions = options.size();

    // Compile the program
    hiprtcResult compileResult = hiprtcCompileProgram(prog, numOptions, options.data());

    // Handle compilation errors
    if(compileResult != HIPRTC_SUCCESS)
    {
        std::size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::string options_str = "";
        for (auto s: options) {
            options_str += std::string(s) + " ";
        }
        FAIL() << "Build Options: " << options_str << "\nCompilation log:\n" << log;
    }

    // Extract compiled code
    std::size_t code_size;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCode(prog, code.data()));

    // Load module and get function
    hipModule_t   module;
    hipFunction_t func;
    CHECK_HIP_ERROR(hipModuleLoadData(&module, code.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func, module, "gemm_rocwmma_d"));

    // Bounds check
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
       || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
    {
        std::cout << "Unsupported size!\n";
        FAIL();
    }

    uint32_t lda = k;
    uint32_t ldb = k;
    uint32_t ldc = n;
    uint32_t ldd = ldc;

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);

    // Allocate and copy device memory
    hipDeviceptr_t d_a, d_b, d_c, d_d, d_d_ref;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesC = matrixC.size() * sizeof(OutputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));
    CHECK_HIP_ERROR(hipMalloc(&d_d_ref, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceil_div(n, ROCWMMA_N * T_BLOCK_Y));

    struct
    {
        uint32_t       _m;
        uint32_t       _n;
        uint32_t       _k;
        hipDeviceptr_t _d_a;
        hipDeviceptr_t _d_b;
        hipDeviceptr_t _d_c;
        hipDeviceptr_t _d_d;
        uint32_t       _lda;
        uint32_t       _ldb;
        uint32_t       _ldc;
        uint32_t       _ldd;
        ComputeT       _alpha;
        ComputeT       _beta;
    } args{m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta};

    std::size_t args_size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size,
                      HIP_LAUNCH_PARAM_END};

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    CHECK_HIP_ERROR(hipModuleLaunchKernel(func,
                                          gridDim.x,
                                          gridDim.y,
                                          gridDim.z,
                                          blockDim.x,
                                          blockDim.y,
                                          blockDim.z,
                                          0,
                                          nullptr,
                                          nullptr,
                                          (void**)&config));

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs" << std::endl;

    std::cout << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    std::vector<OutputT> matrixD_ref(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    rocwmma::gemm_CPU<InputT, OutputT, ComputeT, row_major, col_major, row_major, row_major>(
                    m,
                    n,
                    k,
                    matrixA.data(),
                    matrixB.data(),
                    matrixC.data(),
                    matrixD_ref.data(),
                    alpha,
                    beta);

    CHECK_HIP_ERROR(hipMemcpy(d_d_ref, matrixD_ref.data(), bytesD, hipMemcpyHostToDevice));

    bool result = false;
    double error = 0.0;
    double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;
    std::tie(result, error)
            = rocwmma::compareEqualLaunchKernel<OutputT, OutputT, row_major, row_major>(
                            (OutputT*)d_d, (OutputT*)d_d_ref, m, n, errorTolerance);

    ASSERT_EQ(result, true) << "Result: " << result << "(expecting 1); Error: " << error << " (expecting ~0)\n";

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    CHECK_HIP_ERROR(hipModuleUnload(module));
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    binary_name = std::string(argv[0]);
    return RUN_ALL_TESTS();
}

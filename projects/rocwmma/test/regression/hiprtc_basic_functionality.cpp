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

// HIP RTC (Runtime Compilation) regression tests
//
// These tests validate that the HIP RTC workflow continues to function
// correctly across different HIP/ROCm versions and code changes. They test:
// - Runtime kernel compilation and loading
// - Error handling and diagnostic reporting
// - Module management and kernel execution
// - Support for different data types

#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <string>
#include <vector>

#include <rocwmma/rocwmma.hpp>

#include "regression/reg_common.hpp"

// ============================================================================
// Kernel Sources
// ============================================================================

// Simple vector addition kernel for integer validation
// Note: extern "C" linkage is required for hipModuleGetFunction() to find the kernel
static const char* vectorAddSource = R"(
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

// Intentionally broken kernel source for error handling tests
static const char* brokenKernelSource = R"(
extern "C"
__global__ void brokenKernel(int* data)
{
    // Syntax error: missing semicolon
    int idx = blockIdx.x * blockDim.x + threadIdx.x
}
)";

// Vector multiplication kernel for float validation
static const char* vectorMultiplySource = R"(
extern "C"
__global__ void vectorMultiply(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] * b[idx];
    }
}
)";

// ============================================================================
// Test Suite
// ============================================================================

class HipRTC_BasicFunctionality : public ::testing::Test
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
// Test 1: Basic Compile and Execute
// ============================================================================
// Tests the complete HIP RTC workflow:
// - Create program from source
// - Compile with appropriate options
// - Extract compiled code
// - Load module and get function
// - Launch kernel and verify results
// - Proper resource cleanup

TEST_F(HipRTC_BasicFunctionality, BasicCompileAndExecute)
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
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog, vectorAddSource, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto options    = buildCompileOptions();
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
        FAIL() << "Compilation failed:\n" << log;
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
// Test 2: Compilation Error Handling
// ============================================================================
// Tests that compilation errors are properly detected and reported:
// - Intentional syntax error in kernel
// - Compilation failure is detected
// - Error log is populated with diagnostics
// - No crashes or undefined behavior

TEST_F(HipRTC_BasicFunctionality, CompilationErrorHandling)
{
    // Create program with broken source
    hiprtcProgram prog;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog, brokenKernelSource, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto options    = buildCompileOptions();
    int  numOptions = options.size();

    // Attempt compilation - expect it to fail
    hiprtcResult compileResult = hiprtcCompileProgram(prog, numOptions, options.data());

    // Verify compilation failed
    EXPECT_NE(compileResult, HIPRTC_SUCCESS) << "Compilation should have failed for broken kernel";

    // Get compilation log
    std::size_t log_size;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetProgramLogSize(prog, &log_size));

    // Log should contain error diagnostics
    EXPECT_GT(log_size, 0) << "Compilation log should not be empty for failed compilation";

    if(log_size > 0)
    {
        std::string log(log_size, '\0');
        ASSERT_HIPRTC_SUCCESS(hiprtcGetProgramLog(prog, &log[0]));

        // Verify log contains useful error information
        EXPECT_FALSE(log.empty()) << "Compilation log should contain error diagnostics";
    }

    // Cleanup
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog));
}

// ============================================================================
// Test 3: Multiple Kernel Compilations
// ============================================================================
// Tests that multiple independent kernel compilations work correctly:
// - Compile and execute first kernel
// - Compile and execute second kernel
// - Verify no interference between compilations
// - Test complete cleanup between sessions

TEST_F(HipRTC_BasicFunctionality, MultipleKernelCompilations)
{
    const int n         = 512;
    const int blockSize = 256;
    const int gridSize  = rocwmma::ceil_div(n, blockSize);

    // ========== First Kernel: Vector Add ==========

    // Initialize data for first kernel
    std::vector<int> h_a1(n);
    std::vector<int> h_b1(n);
    std::vector<int> h_c1(n);
    std::vector<int> h_ref1(n);

    for(int i = 0; i < n; i++)
    {
        h_a1[i] = i;
        h_b1[i] = 100 - i;
    }
    vectorAddCPU(h_a1.data(), h_b1.data(), h_ref1.data(), n);

    // Allocate device memory for first kernel
    int* d_a1;
    int* d_b1;
    int* d_c1;
    CHECK_HIP_ERROR(hipMalloc(&d_a1, n * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&d_b1, n * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&d_c1, n * sizeof(int)));
    CHECK_HIP_ERROR(hipMemcpy(d_a1, h_a1.data(), n * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b1, h_b1.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Compile and execute first kernel
    hiprtcProgram prog1;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog1, vectorAddSource, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto options    = buildCompileOptions();
    int  numOptions = options.size();

    // Compile the program
    hiprtcResult compileResult1 = hiprtcCompileProgram(prog1, numOptions, options.data());

    // Handle compilation errors
    if(compileResult1 != HIPRTC_SUCCESS)
    {
        std::size_t log_size;
        hiprtcGetProgramLogSize(prog1, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog1, &log[0]);
        FAIL() << "Compilation failed:\n" << log;
    }

    std::size_t code_size1;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCodeSize(prog1, &code_size1));
    std::vector<char> code1(code_size1);
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCode(prog1, code1.data()));

    hipModule_t   module1;
    hipFunction_t func1;
    CHECK_HIP_ERROR(hipModuleLoadData(&module1, code1.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func1, module1, "vectorAdd"));

    struct
    {
        int* d_a;
        int* d_b;
        int* d_c;
        int  n;
    } args1{d_a1, d_b1, d_c1, n};

    std::size_t args_size1 = sizeof(args1);
    void*       config1[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                              &args1,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                              &args_size1,
                              HIP_LAUNCH_PARAM_END};

    CHECK_HIP_ERROR(hipModuleLaunchKernel(func1, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, nullptr, (void**)&config1));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipMemcpy(h_c1.data(), d_c1, n * sizeof(int), hipMemcpyDeviceToHost));

    // Verify first kernel results
    for(int i = 0; i < n; i++)
    {
        EXPECT_EQ(h_c1[i], h_ref1[i]) << "First kernel: Mismatch at index " << i;
    }

    // Cleanup first kernel
    CHECK_HIP_ERROR(hipModuleUnload(module1));
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog1));
    CHECK_HIP_ERROR(hipFree(d_a1));
    CHECK_HIP_ERROR(hipFree(d_b1));
    CHECK_HIP_ERROR(hipFree(d_c1));

    // ========== Second Kernel: Vector Multiply ==========

    // Initialize data for second kernel
    std::vector<float> h_a2(n);
    std::vector<float> h_b2(n);
    std::vector<float> h_c2(n);
    std::vector<float> h_ref2(n);

    for(int i = 0; i < n; i++)
    {
        h_a2[i] = static_cast<float>(i) * 0.5f;
        h_b2[i] = 2.0f;
    }
    vectorMultiplyCPU(h_a2.data(), h_b2.data(), h_ref2.data(), n);

    // Allocate device memory for second kernel
    float* d_a2;
    float* d_b2;
    float* d_c2;
    CHECK_HIP_ERROR(hipMalloc(&d_a2, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_b2, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_c2, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_a2, h_a2.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b2, h_b2.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // Compile and execute second kernel
    hiprtcProgram prog2;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog2, vectorMultiplySource, nullptr, 0, nullptr, nullptr));
    // Compile the program
    hiprtcResult compileResult2 = hiprtcCompileProgram(prog2, numOptions, options.data());

    // Handle compilation errors
    if(compileResult2 != HIPRTC_SUCCESS)
    {
        std::size_t log_size;
        hiprtcGetProgramLogSize(prog2, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog2, &log[0]);
        FAIL() << "Compilation failed:\n" << log;
    }


    std::size_t code_size2;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCodeSize(prog2, &code_size2));
    std::vector<char> code2(code_size2);
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCode(prog2, code2.data()));

    hipModule_t   module2;
    hipFunction_t func2;
    CHECK_HIP_ERROR(hipModuleLoadData(&module2, code2.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func2, module2, "vectorMultiply"));

    struct
    {
        float* d_a;
        float* d_b;
        float* d_c;
        int    n;
    } args2{d_a2, d_b2, d_c2, n};

    std::size_t args_size2 = sizeof(args2);
    void*       config2[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                              &args2,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                              &args_size2,
                              HIP_LAUNCH_PARAM_END};

    CHECK_HIP_ERROR(hipModuleLaunchKernel(func2, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, nullptr, (void**)&config2));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipMemcpy(h_c2.data(), d_c2, n * sizeof(float), hipMemcpyDeviceToHost));

    // Verify second kernel results with floating-point tolerance
    for(int i = 0; i < n; i++)
    {
        EXPECT_NEAR(h_c2[i], h_ref2[i], 1e-5f) << "Second kernel: Mismatch at index " << i;
    }

    // Cleanup second kernel
    CHECK_HIP_ERROR(hipModuleUnload(module2));
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog2));
    CHECK_HIP_ERROR(hipFree(d_a2));
    CHECK_HIP_ERROR(hipFree(d_b2));
    CHECK_HIP_ERROR(hipFree(d_c2));
}

// ============================================================================
// Test 4: Float Type Test
// ============================================================================
// Tests HIP RTC with floating-point data types:
// - Compile kernel using float types
// - Execute and verify with appropriate tolerance
// - Validate floating-point operations

TEST_F(HipRTC_BasicFunctionality, FloatTypeTest)
{
    const int n         = 1024;
    const int blockSize = 256;
    const int gridSize  = rocwmma::ceil_div(n, blockSize);

    // Initialize host data
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);
    std::vector<float> h_reference(n);

    for(int i = 0; i < n; i++)
    {
        h_a[i] = static_cast<float>(i) * 0.1f;
        h_b[i] = static_cast<float>(i) * 0.2f;
    }

    vectorMultiplyCPU(h_a.data(), h_b.data(), h_reference.data(), n);

    // Allocate device memory
    float* d_a;
    float* d_b;
    float* d_c;
    CHECK_HIP_ERROR(hipMalloc(&d_a, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // Create and compile program
    hiprtcProgram prog;
    ASSERT_HIPRTC_SUCCESS(hiprtcCreateProgram(&prog, vectorMultiplySource, nullptr, 0, nullptr, nullptr));

    // Build compile options
    auto options    = buildCompileOptions();
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
        FAIL() << "Compilation failed:\n" << log;
    }

    // Extract code
    std::size_t code_size;
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);
    ASSERT_HIPRTC_SUCCESS(hiprtcGetCode(prog, code.data()));

    // Load and execute
    hipModule_t   module;
    hipFunction_t func;
    CHECK_HIP_ERROR(hipModuleLoadData(&module, code.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func, module, "vectorMultiply"));

    struct
    {
        float* d_a;
        float* d_b;
        float* d_c;
        int    n;
    } args{d_a, d_b, d_c, n};

    std::size_t args_size = sizeof(args);
    void*       config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                             &args,
                             HIP_LAUNCH_PARAM_BUFFER_SIZE,
                             &args_size,
                             HIP_LAUNCH_PARAM_END};

    CHECK_HIP_ERROR(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, nullptr, (void**)&config));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipMemcpy(h_c.data(), d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    // Verify with floating-point tolerance
    for(int i = 0; i < n; i++)
    {
        EXPECT_NEAR(h_c[i], h_reference[i], 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    CHECK_HIP_ERROR(hipModuleUnload(module));
    ASSERT_HIPRTC_SUCCESS(hiprtcDestroyProgram(&prog));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
}

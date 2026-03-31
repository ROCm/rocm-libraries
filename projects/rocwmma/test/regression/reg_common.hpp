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

#include <filesystem>
#include "common.hpp"
#include "reference.hpp"

#ifndef ROCWMMA_TEST_REGRESSION_COMMON_HPP
#define ROCWMMA_TEST_REGRESSION_COMMON_HPP

// GTest-compatible error handling macros for HIP RTC
// Unlike the samples/common.hpp CHECK_HIPRTC_ERROR macro which calls exit(),
// these macros use GTest's ASSERT/EXPECT to allow proper test failure reporting

#define PRINT_COMPILE_LOG(hiprtcProgram)                                              \
    do                                                                                \
    {                                                                                 \
        size_t logSize;                                                               \
        hiprtcGetProgramLogSize(hiprtcProgram, &logSize);                             \
        if (logSize) {                                                                \
            std::string log(logSize, '\0');                                           \
            hiprtcGetProgramLog(hiprtcProgram, &log[0]);                              \
            std::cout << log << std::endl;                                            \
        }                                                                             \
    } while(0)

#define ASSERT_HIPRTC_SUCCESS_HELPER(hiprtcProgram, expression)                              \
    do                                                                                \
    {                                                                                 \
        hiprtcResult status = (expression);                                           \
        if (hiprtcProgram && status != HIPRTC_SUCCESS) PRINT_COMPILE_LOG(hiprtcProgram);               \
        ASSERT_EQ(status, HIPRTC_SUCCESS) << "hipRTC error: " << hiprtcGetErrorString(status) \
                                          << " at " << __FILE__ << ":" << __LINE__;   \
    } while(0)

#define EXPECT_HIPRTC_SUCCESS_HELPER(hiprtcProgram, expression)                              \
    do                                                                                \
    {                                                                                 \
        hiprtcResult status = (expression);                                           \
        if (hiprtcProgram && status != HIPRTC_SUCCESS) PRINT_COMPILE_LOG(hiprtcProgram);               \
        EXPECT_EQ(status, HIPRTC_SUCCESS) << "hipRTC error: " << hiprtcGetErrorString(status) \
                                          << " at " << __FILE__ << ":" << __LINE__;   \
    } while(0)

#define ASSERT_HIPRTC_SUCCESS_COMPILE(hiprtcProgram, expression)                              \
        ASSERT_HIPRTC_SUCCESS_HELPER(hiprtcProgram, expression)

#define ASSERT_HIPRTC_SUCCESS(expression) \
        ASSERT_HIPRTC_SUCCESS_HELPER(nullptr, expression)

#define EXPECT_HIPRTC_SUCCESS_COMPILE(hiprtcProgram, expression)                              \
        EXPECT_HIPRTC_SUCCESS_HELPER(hiprtcProgram, expression)

#define EXPECT_HIPRTC_SUCCESS(expression) \
        EXPECT_HIPRTC_SUCCESS_HELPER(nullptr, expression)

// ============================================================================
// Helper Functions
// ============================================================================

static inline std::string getIncludeDirArg(const std::string &binary_name) {
    auto binary_path = std::filesystem::canonical(binary_name).parent_path();
    try {
        auto include_path = std::filesystem::canonical(binary_path.string() + "/../include");
        return "-I" + include_path.string();
    } catch(const std::filesystem::filesystem_error&) {
        return "";
    }
}

// Get ROCM installation path from environment or use default
static inline std::string getRocwmmaIncludePath()
{
    const char* rocwmmaPath = std::getenv("ROCWMMA_INCLUDE_PATH");
    if (rocwmmaPath)
        return std::string(rocwmmaPath);
    const char* rocmPath = std::getenv("ROCM_PATH");
    return rocmPath ? std::string(rocmPath) + "/include" : "/opt/rocm/include";
}

// Build standard compilation options for HIP RTC
static inline std::vector<const char*> buildCompileOptions()
{
    static std::string rocwmmaIncludePath = "-I" + getRocwmmaIncludePath();

    std::vector<const char*> options;
    options.push_back("-D__HIP_PLATFORM_AMD__");
    options.push_back("--std=c++17");
    options.push_back(rocwmmaIncludePath.c_str());
    options.push_back("-I/opt/rocm/include");

    return options;
}

// CPU reference implementation for vector addition
static inline void vectorAddCPU(const int* a, const int* b, int* c, int n)
{
    for(int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// CPU reference implementation for vector multiplication
static inline void vectorMultiplyCPU(const float* a, const float* b, float* c, int n)
{
    for(int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

// Host matrix data random initialization
template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };
    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

// HIP Host function to retrieve the warp size
enum hipWarpSize_t : uint32_t
{
    Wave32 = 32,
    Wave64 = 64,
    UNSUPPORTED_WARP_SIZE,
};

static inline uint32_t getWarpSize()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;
    uint32_t        mWarpSize = hipWarpSize_t::UNSUPPORTED_WARP_SIZE;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    switch(mProps.warpSize)
    {
    case hipWarpSize_t::Wave32:
    case hipWarpSize_t::Wave64:
        mWarpSize = mProps.warpSize;
    default:;
    }

    if(mWarpSize == hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
    {
        std::cerr << "Cannot proceed: unsupported warp sizev detected. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    return mWarpSize;
}

#endif // ROCWMMA_TEST_REGRESSION_COMMON_HPP

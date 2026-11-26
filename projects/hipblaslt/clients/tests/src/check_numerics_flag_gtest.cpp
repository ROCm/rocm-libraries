// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// HIPBLASLT_CHECK_NUMERICS: the flag the NaN scanner claims with atomicCAS
// must be device memory. An atomic on mapped host memory is a PCIe AtomicOp,
// which a root port without AtomicOp support drops, and then the first NaN is
// never reported. The end-to-end cases in testing_aux_check_numerics_gemm.hpp
// fail only on a GPU behind such a root port, and CI has none, but the flag's
// memory type can be checked on any GPU. The `smoke` token puts the test in
// the quick tier that PR CI runs.

#include "scoped_env.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <string>

extern "C" void* hipblaslt_check_numerics_flag_for_test(void* handle);

namespace
{
    using test_env::ScopedEnv;

    TEST(CheckNumerics, smoke_FlagIsDeviceMemory)
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
            GTEST_SKIP() << "No GPU available";

        // The handle reads these when it is created.
        ScopedEnv check("HIPBLASLT_CHECK_NUMERICS", "1");
        for(const char* stopOnFirst : {"0", "1"})
        {
            SCOPED_TRACE(std::string("HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST=") + stopOnFirst);
            ScopedEnv stop("HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST", stopOnFirst);

            hipblasLtHandle_t handle = nullptr;
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            const void* flag = hipblaslt_check_numerics_flag_for_test(handle);
            EXPECT_NE(flag, nullptr);
            if(flag)
            {
                hipPointerAttribute_t attr{};
                EXPECT_EQ(hipPointerGetAttributes(&attr, flag), hipSuccess);
                EXPECT_EQ(attr.type, hipMemoryTypeDevice)
                    << "the scanner's atomicCAS must not target host memory";
            }
            EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }
    }
} // namespace

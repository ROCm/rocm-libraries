// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A public function that fails with a C++ exception reports the exception text through the
// error log, so the cause is visible beside the status the caller receives.

#include "../src/scoped_env.hpp"

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>

namespace
{
    using test_env::ScopedEnv;

    // The unparsable override makes std::stoi throw inside hipblasLtMatmulDescCreate.
    void createDescWithBadOverrideAndExit()
    {
        ScopedEnv override("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", "not-a-number");

        hipblasLtMatmulDesc_t desc = nullptr;
        auto status = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F_FAST_TF32, HIP_R_32F);
        if(status != HIPBLAS_STATUS_INTERNAL_ERROR)
            std::exit(1);
        std::exit(0);
    }

    TEST(HostUnitExceptionLoggingDeathTest, smoke_ExceptionMessageIsLogged)
    {
        // The logger reads its environment once per process, so the case runs in a newly
        // started copy of this binary and the test matches the copy's stderr.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv level("HIPBLASLT_LOG_LEVEL", "1");
        ScopedEnv file("HIPBLASLT_LOG_FILE", nullptr);

        EXPECT_EXIT(createDescWithBadOverrideAndExit(),
                    ::testing::ExitedWithCode(0),
                    "exception_to_hipblas_status.*stoi");
    }
} // namespace

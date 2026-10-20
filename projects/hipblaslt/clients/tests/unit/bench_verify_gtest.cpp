// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the --verify_strict exit-code rule in hipblaslt_bench_options.
// See README.md in this directory for the HostUnit/smoke_ naming.

#include "hipblaslt_bench_options.hpp"

#include <gtest/gtest.h>

namespace
{
    class HostUnitBenchVerify : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            saved_strict   = hipblaslt_bench_options::verify_strict();
            saved_failures = hipblaslt_bench_options::verification_failures();
        }

        void TearDown() override
        {
            hipblaslt_bench_options::verify_strict()         = saved_strict;
            hipblaslt_bench_options::verification_failures() = saved_failures;
        }

        static int exit_code(bool strict, int64_t failures, int ret)
        {
            hipblaslt_bench_options::verify_strict()         = strict;
            hipblaslt_bench_options::verification_failures() = failures;
            return hipblaslt_bench_options::verification_exit_code(ret);
        }

        bool    saved_strict   = false;
        int64_t saved_failures = 0;
    };

    TEST_F(HostUnitBenchVerify, smoke_StrictWithFailureExitsTwo)
    {
        EXPECT_EQ(exit_code(true, 1, 0), 2);
        EXPECT_EQ(exit_code(true, 7, 0), 2);
    }

    TEST_F(HostUnitBenchVerify, smoke_DefaultKeepsExitZeroOnFailure)
    {
        EXPECT_EQ(exit_code(false, 3, 0), 0);
    }

    TEST_F(HostUnitBenchVerify, smoke_StrictWithoutFailureExitsZero)
    {
        EXPECT_EQ(exit_code(true, 0, 0), 0);
    }

    // A run that already failed for another reason keeps its own exit code.
    TEST_F(HostUnitBenchVerify, smoke_EarlierErrorKeepsItsCode)
    {
        EXPECT_EQ(exit_code(true, 2, 1), 1);
        EXPECT_EQ(exit_code(true, 0, -1), -1);
        EXPECT_EQ(exit_code(false, 0, 1), 1);
    }
}

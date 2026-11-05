// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "client_reporting.hpp"
#include <gtest/gtest-spi.h>

namespace
{
    TEST(HostUnitClientReporting, smoke_FatalFailureReturnsFromCaller)
    {
        ::testing::TestPartResultArray results;
        int                            evaluations = 0;
        bool                           continued   = false;
        {
            ::testing::ScopedFakeTestPartResultReporter capture(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD,
                &results);
            [&] {
                for(int i = 0; i < 2; ++i)
                    HIPBLASLT_FAIL("failure " << ++evaluations);
                continued = true;
            }();
        }
        EXPECT_FALSE(continued);
        EXPECT_EQ(evaluations, 1);
        ASSERT_EQ(results.size(), 1);
        EXPECT_TRUE(results.GetTestPartResult(0).fatally_failed());
        EXPECT_STREQ(results.GetTestPartResult(0).file_name(), __FILE__);
        EXPECT_NE(std::string(results.GetTestPartResult(0).message()).find("failure 1"),
                  std::string::npos);
    }

    TEST(HostUnitClientReporting, smoke_SkipReturnsFromCaller)
    {
        ::testing::TestPartResultArray results;
        bool                           continued = false;
        {
            ::testing::ScopedFakeTestPartResultReporter capture(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD,
                &results);
            [&] {
                HIPBLASLT_SKIP("unsupported case");
                continued = true;
            }();
        }
        EXPECT_FALSE(continued);
        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results.GetTestPartResult(0).type(), ::testing::TestPartResult::kSkip);
    }

    TEST(HostUnitClientReporting, smoke_NonfatalFailureContinues)
    {
        bool continued = false;
        EXPECT_NONFATAL_FAILURE(
            {
                HIPBLASLT_REPORT_FAILURE("validation failed");
                continued = true;
            },
            "validation failed");
        EXPECT_TRUE(continued);
    }
}

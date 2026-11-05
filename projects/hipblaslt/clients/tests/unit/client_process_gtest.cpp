// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "client_process.hpp"
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace
{
    class HostUnitClientProcess : public ::testing::Test
    {
    protected:
        static constexpr const char* name = "HIPBLASLT_TEST_PROCESS_ENV";
        std::optional<std::string>   saved;

        void SetUp() override
        {
            if(const char* value = std::getenv(name))
                saved = value;
            ASSERT_EQ(hipblaslt_clients::unsetEnv(name), 0);
        }

        void TearDown() override
        {
            if(saved)
                EXPECT_EQ(hipblaslt_clients::setEnv(name, saved->c_str(), 1), 0);
            else
                EXPECT_EQ(hipblaslt_clients::unsetEnv(name), 0);
        }
    };

    TEST_F(HostUnitClientProcess, smoke_EnvironmentOverwriteAndRemoval)
    {
        ASSERT_EQ(hipblaslt_clients::setEnv(name, "first", 0), 0);
        EXPECT_STREQ(std::getenv(name), "first");
        ASSERT_EQ(hipblaslt_clients::setEnv(name, "ignored", 0), 0);
        EXPECT_STREQ(std::getenv(name), "first");
        ASSERT_EQ(hipblaslt_clients::setEnv(name, "second", 1), 0);
        EXPECT_STREQ(std::getenv(name), "second");
        ASSERT_EQ(hipblaslt_clients::unsetEnv(name), 0);
        EXPECT_EQ(std::getenv(name), nullptr);
        EXPECT_EQ(hipblaslt_clients::unsetEnv(name), 0);
    }
}

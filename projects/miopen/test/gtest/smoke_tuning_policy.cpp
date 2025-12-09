// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/miopen.h>

#include "get_handle.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gtest_common.hpp"
#include "miopen/db_path.hpp"
#include "../lib_env_var.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_USER_DB_PATH)
MIOPEN_LIB_ENV_VAR(MIOPEN_ENABLE_LOGGING)

class CPU_TuningPolicy_NONE : public ::testing::Test
{
protected:
    void testSetInvalidValue(const miopenTuningPolicy_t original_policy, int invalid)
    {
        auto&& handle                           = get_handle();
        miopenTuningPolicy_t test_tuning_policy = miopenTuningPolicy_t::miopenTuningPolicyNone;
        miopenTuningPolicy_t prev_tuning_policy;

        EXPECT_EQ(miopenSetTuningPolicy(&handle, original_policy), miopenStatusSuccess);
        EXPECT_EQ(miopenGetTuningPolicy(&handle, &prev_tuning_policy), miopenStatusSuccess);
        EXPECT_EQ(original_policy, prev_tuning_policy);

        EXPECT_EQ(miopenSetTuningPolicy(&handle, static_cast<miopenTuningPolicy_t>(invalid)),
                  miopenStatusBadParm);
        EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
        EXPECT_EQ(prev_tuning_policy, test_tuning_policy);
    }
};

TEST_F(CPU_TuningPolicy_NONE, TestTuningPolicyGetterAndSetterValidValues)
{
    auto&& handle = get_handle();
    miopenTuningPolicy_t test_tuning_policy;

    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyNone);

    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate);

    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicySearch),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicySearch);

    EXPECT_EQ(miopenSetTuningPolicy(&handle, static_cast<miopenTuningPolicy_t>(4)),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicySearchDbUpdate);

    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyDbClean),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyDbClean);

    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyNone),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyNone);
}

TEST_F(CPU_TuningPolicy_NONE, TestTuningPolicyGetterAndSetterErrorHandling)
{
    testSetInvalidValue(miopenTuningPolicy_t::miopenTuningPolicyDbUpdate, -1);
    testSetInvalidValue(miopenTuningPolicy_t::miopenTuningPolicySearch, 0);
    testSetInvalidValue(miopenTuningPolicy_t::miopenTuningPolicySearchDbUpdate, 6);
    testSetInvalidValue(miopenTuningPolicy_t::miopenTuningPolicyDbClean, 1000);
}

TEST_F(CPU_TuningPolicy_NONE, TestNullHandleForTuningPolicy)
{
    miopenTuningPolicy_t test_tuning_policy;

    EXPECT_EQ(miopenGetTuningPolicy(nullptr, &test_tuning_policy), miopenStatusBadParm);

    EXPECT_EQ(miopenSetTuningPolicy(nullptr, miopenTuningPolicy_t::miopenTuningPolicyNone),
              miopenStatusBadParm);
}

TEST_F(CPU_TuningPolicy_NONE, TestNullPolicyPointer)
{
    auto&& handle = get_handle();

    EXPECT_EQ(miopenGetTuningPolicy(&handle, nullptr), miopenStatusBadParm);
}

TEST_F(CPU_TuningPolicy_NONE, TestGetApiLogged)
{
    ScopedEnvironment<std::string> enable_logging(MIOPEN_ENABLE_LOGGING, "ON");
    miopenTuningPolicy_t policy;
    auto&& handle = get_handle();

    testing::internal::CaptureStderr();
    auto status = miopenGetTuningPolicy(&handle, &policy);
    ASSERT_EQ(status, miopenStatusSuccess);
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(output, testing::HasSubstr(" miopenGetTuningPolicy("));
}

TEST_F(CPU_TuningPolicy_NONE, TestSetApiLogged)
{
    miopenTuningPolicy_t original_policy;
    auto&& handle = get_handle();

    auto status = miopenGetTuningPolicy(&handle, &original_policy);
    ASSERT_EQ(status, miopenStatusSuccess);

    {
        ScopedEnvironment<std::string> enable_logging(MIOPEN_ENABLE_LOGGING, "ON");

        testing::internal::CaptureStderr();
        status = miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicySearch);
        ASSERT_EQ(status, miopenStatusSuccess);
        std::string output = testing::internal::GetCapturedStderr();
        EXPECT_THAT(output, testing::HasSubstr(" miopenSetTuningPolicy("));
    }

    status = miopenSetTuningPolicy(&handle, original_policy);
    ASSERT_EQ(status, miopenStatusSuccess);
}

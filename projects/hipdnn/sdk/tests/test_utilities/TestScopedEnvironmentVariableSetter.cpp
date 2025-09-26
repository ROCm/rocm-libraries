// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

using namespace hipdnn_sdk::test_utilities;

TEST(TestScopedEnvironmentVariableSetter, RestoresOriginalValue)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR";
    const char* originalValue = "original";
    const char* newValue = "modified";

    hipdnn_sdk::utilities::setEnv(testVar, originalValue);

    {
        ScopedEnvironmentVariableSetter guard(testVar, newValue);

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), newValue);
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), originalValue);

    hipdnn_sdk::utilities::unsetEnv(testVar);
}

TEST(TestScopedEnvironmentVariableSetter, RestoresUnsetVariable)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR_UNSET";

    hipdnn_sdk::utilities::unsetEnv(testVar);

    {
        ScopedEnvironmentVariableSetter guard(testVar, "temporary");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "temporary");
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "");
    EXPECT_TRUE(hipdnn_sdk::utilities::getEnv(testVar).empty());
}

TEST(TestScopedEnvironmentVariableSetter, HandlesEmptyValue)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR_EMPTY";

    hipdnn_sdk::utilities::setEnv(testVar, "");

    {
        ScopedEnvironmentVariableSetter guard(testVar, "non-empty");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "non-empty");
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "");

    hipdnn_sdk::utilities::unsetEnv(testVar);
}

TEST(TestScopedEnvironmentVariableSetter, SetValueMethod)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR_SETVALUE";
    const char* originalValue = "original";

    hipdnn_sdk::utilities::setEnv(testVar, originalValue);

    {
        ScopedEnvironmentVariableSetter guard(testVar, "initial");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "initial");

        guard.setValue("updated");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "updated");
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), originalValue);

    hipdnn_sdk::utilities::unsetEnv(testVar);
}

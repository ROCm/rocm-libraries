// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestAutotuneCacheEnv.cpp
 * @brief Contract tests for the disable-env-var helper
 *        (`exactCacheDisabled`, `HIPDNN_DISABLE_EXACT_ENGINE_CACHE`).
 */

#include "heuristics/config/AutotuneCacheEnv.hpp"

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <gtest/gtest.h>

using namespace hipdnn_backend::heuristics::config;

namespace
{
constexpr const char* DISABLE_ENV = "HIPDNN_DISABLE_EXACT_ENGINE_CACHE";
} // namespace

TEST(TestAutotuneCacheEnv, UnsetReturnsFalse)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV);

    EXPECT_FALSE(exactCacheDisabled());
}

TEST(TestAutotuneCacheEnv, TruthyValueReturnsTrue)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV, "1");

    EXPECT_TRUE(exactCacheDisabled());
}

TEST(TestAutotuneCacheEnv, EmptyOrWhitespaceOnlyReturnsFalse)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV, "   ");

    EXPECT_FALSE(exactCacheDisabled());
}

// Presence alone must not disable the cache; the value is checked against a literal set.
TEST(TestAutotuneCacheEnv, ExplicitlyFalsyValuesLeaveCacheEnabled)
{
    for(const char* value : {"0", "false", "off", "no", "disable", "disabled"})
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV, value);

        EXPECT_FALSE(exactCacheDisabled()) << "value=" << value;
    }
}

TEST(TestAutotuneCacheEnv, TruthySpellingsAreAccepted)
{
    for(const char* value : {"1", "true", "on", "yes", "enable", "enabled", "TRUE", " on "})
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV, value);

        EXPECT_TRUE(exactCacheDisabled()) << "value=" << value;
    }
}

TEST(TestAutotuneCacheEnv, UnrecognizedValueLeavesCacheEnabled)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(DISABLE_ENV, "maybe");

    EXPECT_FALSE(exactCacheDisabled());
}

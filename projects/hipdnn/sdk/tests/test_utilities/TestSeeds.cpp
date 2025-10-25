// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <unordered_set>

using namespace hipdnn_sdk::test_utilities;

class TestSeeds : public ::testing::Test
{
protected:
    std::unique_ptr<ScopedEnvironmentVariableSetter> _seedGuard;

    void SetUp() override
    {
        _seedGuard = std::make_unique<ScopedEnvironmentVariableSetter>("HIPDNN_GLOBAL_TEST_SEED");
    }

    void TearDown() override
    {
        _seedGuard.reset();
    }
};

TEST_F(TestSeeds, ReturnsDefaultSeedWhenEnvVarNotSet)
{
    hipdnn_sdk::utilities::unsetEnv("HIPDNN_GLOBAL_TEST_SEED");

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 1u) << "Expected default seed of 1 when environment variable is not set";
}

TEST_F(TestSeeds, ReturnsDefaultSeedWhenEnvVarIsEmpty)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "");

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 1u) << "Expected default seed of 1 when environment variable is empty";
}

TEST_F(TestSeeds, ReturnsSpecificSeedWhenNumericValueProvided)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "42");

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 42u) << "Expected seed to match numeric value from environment variable";
}

TEST_F(TestSeeds, HandlesLargeNumericValues)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "4294967295"); // Max uint32

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 4294967295u) << "Expected seed to handle maximum unsigned int value";
}

TEST_F(TestSeeds, ReturnsRandomSeedWhenRandomSpecified)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "RANDOM");

    std::unordered_set<unsigned int> seeds;
    constexpr int ITERATIONS = 100;

    for(int i = 0; i < ITERATIONS; ++i)
    {
        seeds.insert(getGlobalTestSeed());
    }

    EXPECT_GT(seeds.size(), 1u)
        << "Expected multiple unique seeds when RANDOM is specified, but got the same seed "
        << ITERATIONS << " times";
}

TEST_F(TestSeeds, RandomIsCaseInsensitive)
{
    std::vector<std::string> randomVariants = {"RANDOM", "random", "Random", "RaNdOm"};

    for(const auto& variant : randomVariants)
    {
        hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", variant.c_str());

        std::unordered_set<unsigned int> seeds;
        constexpr int ITERATIONS = 100;

        for(int i = 0; i < ITERATIONS; ++i)
        {
            seeds.insert(getGlobalTestSeed());
        }

        EXPECT_GT(seeds.size(), 1u)
            << "Expected multiple unique seeds when RANDOM is specified, but got the same seed "
            << ITERATIONS << " times";
    }
}

TEST_F(TestSeeds, ReturnsDefaultSeedForInvalidNumericValue)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "not_a_number");

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 1u)
        << "Expected default seed of 1 when environment variable contains invalid numeric value";
}

TEST_F(TestSeeds, ReturnsDefaultSeedForOutOfRangeValue)
{
    // Value larger than unsigned long can hold
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "99999999999999999999999999");

    auto seed = getGlobalTestSeed();

    EXPECT_EQ(seed, 1u)
        << "Expected default seed of 1 when environment variable exceeds valid range";
}

TEST_F(TestSeeds, ConsistentSeedProducesSameValue)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_GLOBAL_TEST_SEED", "12345");

    auto seed1 = getGlobalTestSeed();
    auto seed2 = getGlobalTestSeed();
    auto seed3 = getGlobalTestSeed();

    EXPECT_EQ(seed1, 12345u);
    EXPECT_EQ(seed2, 12345u);
    EXPECT_EQ(seed3, 12345u);
}

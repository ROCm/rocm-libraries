// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/ModuleCache.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace
{

// Trivial value type for testing — a shared_ptr to an int.
using TestValue = std::shared_ptr<int>;

// A mock derived class that uses the CRTP base.  The load function
// returns a TestValue constructed from the key, or nullptr if the key
// starts with "FAIL".
class MockModuleCache : public hip_kernel_provider::compilation::
                            ModuleCache<MockModuleCache, TestValue, const std::string&>
{
public:
    static std::string makeKey(const std::string& name)
    {
        return name;
    }

    static TestValue load(const std::string& name)
    {
        if(name.rfind("FAIL", 0) == 0)
        {
            return nullptr;
        }
        // Use the key length as a distinguishable payload.
        return std::make_shared<int>(static_cast<int>(name.size()));
    }
};

TEST(TestModuleCache, EmptyOnConstruction)
{
    const MockModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestModuleCache, GetOrLoadCallsLoadOnMiss)
{
    MockModuleCache cache;
    auto result = cache.getOrLoad("kernel_a");

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, static_cast<int>(std::string("kernel_a").size()));
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains("kernel_a"));
}

TEST(TestModuleCache, GetOrLoadReturnsCachedOnHit)
{
    MockModuleCache cache;
    auto first = cache.getOrLoad("kernel_b");
    auto second = cache.getOrLoad("kernel_b");

    // Must be the exact same object (pointer equality).
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(cache.size(), 1u);
}

TEST(TestModuleCache, FailedLoadNotCached)
{
    MockModuleCache cache;
    auto result = cache.getOrLoad("FAIL_kernel");

    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains("FAIL_kernel"));
}

TEST(TestModuleCache, ContainsAndSizeTracking)
{
    MockModuleCache cache;
    EXPECT_FALSE(cache.contains("x"));

    cache.getOrLoad("x");
    cache.getOrLoad("y");
    cache.getOrLoad("FAIL_z");

    EXPECT_TRUE(cache.contains("x"));
    EXPECT_TRUE(cache.contains("y"));
    EXPECT_FALSE(cache.contains("FAIL_z"));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(TestModuleCache, SeparateInstancesAreIsolated)
{
    MockModuleCache cacheA;
    const MockModuleCache cacheB;

    cacheA.getOrLoad("shared_key");

    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 0u);
    EXPECT_TRUE(cacheA.contains("shared_key"));
    EXPECT_FALSE(cacheB.contains("shared_key"));
}

} // namespace

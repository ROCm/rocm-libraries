// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/EnginePluginMacros.hpp>

using namespace hipdnn_plugin_sdk;

namespace
{

// Test container that registers a mock engine
class TestPluginContainer : public EnginePluginContainer
{
public:
    TestPluginContainer() = default;
};

// Test handle using the base implementation
struct TestPluginHandle : public PluginHandleBase<TestPluginContainer>
{
};

// For the execution context, we use the base implementation
// In a real plugin, you might extend this

} // namespace

// Note: We can't actually use the macro in tests because it defines
// the actual C API functions which would conflict with the real plugin.
// Instead, we test the supporting classes.

TEST(TestPluginHandleBase, InitialStreamIsNull)
{
    TestPluginHandle handle;
    EXPECT_EQ(handle.getStream(), nullptr);
}

TEST(TestPluginHandleBase, SetStreamUpdatesStream)
{
    TestPluginHandle handle;
    auto testStream = reinterpret_cast<hipStream_t>(0x12345678);
    handle.setStream(testStream);
    EXPECT_EQ(handle.getStream(), testStream);
}

TEST(TestPluginHandleBase, ContainerAccessWorks)
{
    TestPluginHandle handle;
    handle.container = std::make_shared<TestPluginContainer>();

    // Should be able to get the engine manager without crashing
    auto& engineManager = handle.getEngineManager();
    (void)engineManager;

    // Engine manager should be empty since we didn't register any engines
    EXPECT_TRUE(handle.getEngineManager().getAllEngineIds().empty());
}

// Test that the macro header can be included and compiles correctly
// by verifying the types are properly defined
TEST(TestEnginePluginMacros, PluginHandleBaseIsDefined)
{
    // This test simply verifies that PluginHandleBase template compiles
    using HandleType = PluginHandleBase<EnginePluginContainer>;
    HandleType handle;
    EXPECT_EQ(handle.stream, nullptr);
}

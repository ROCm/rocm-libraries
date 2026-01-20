// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include <hipdnn_plugin_sdk/EnginePluginContainer.hpp>

using namespace hipdnn_plugin_sdk;

namespace
{

// Test container that tracks initialization
class TestContainer : public EnginePluginContainer
{
public:
    TestContainer()
    {
        instanceCount++;
    }

    ~TestContainer() override
    {
        instanceCount--;
    }

    static int instanceCount;
};

int TestContainer::instanceCount = 0;

} // namespace

TEST(TestEnginePluginContainer, HasEngineManager)
{
    EnginePluginContainer container;
    // Just verify we can get the engine manager without crashing
    auto& manager = container.getEngineManager();
    (void)manager;
}

TEST(TestEnginePluginContainer, EngineManagerInitiallyEmpty)
{
    EnginePluginContainer container;
    auto engineIds = container.getEngineManager().getAllEngineIds();
    EXPECT_TRUE(engineIds.empty());
}

TEST(TestSharedContainerManager, CreatesContainerOnFirstCall)
{
    TestContainer::instanceCount = 0;
    SharedContainerManager<TestContainer> manager;

    EXPECT_EQ(TestContainer::instanceCount, 0);

    auto container = manager.getOrCreate();
    EXPECT_NE(container, nullptr);
    EXPECT_EQ(TestContainer::instanceCount, 1);
}

TEST(TestSharedContainerManager, ReturnsSameContainerOnSubsequentCalls)
{
    TestContainer::instanceCount = 0;
    SharedContainerManager<TestContainer> manager;

    auto container1 = manager.getOrCreate();
    auto container2 = manager.getOrCreate();

    EXPECT_EQ(container1.get(), container2.get());
    EXPECT_EQ(TestContainer::instanceCount, 1);
}

TEST(TestSharedContainerManager, RecreatesContainerAfterAllReferencesDropped)
{
    TestContainer::instanceCount = 0;
    SharedContainerManager<TestContainer> manager;

    {
        auto container = manager.getOrCreate();
        EXPECT_EQ(TestContainer::instanceCount, 1);
    }

    // Container should be destroyed now
    EXPECT_EQ(TestContainer::instanceCount, 0);

    // Getting again should create a new one
    auto newContainer = manager.getOrCreate();
    EXPECT_EQ(TestContainer::instanceCount, 1);
}

TEST(TestSharedContainerManager, ThreadSafeCreation)
{
    TestContainer::instanceCount = 0;
    SharedContainerManager<TestContainer> manager;

    std::vector<std::shared_ptr<TestContainer>> containers;
    std::mutex containersMutex;

    constexpr int NUM_THREADS = 10;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for(int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            auto container = manager.getOrCreate();
            std::lock_guard<std::mutex> lock(containersMutex);
            containers.push_back(container);
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    // All threads should have gotten the same container
    EXPECT_EQ(TestContainer::instanceCount, 1);
    EXPECT_EQ(containers.size(), static_cast<size_t>(NUM_THREADS));

    // Verify all containers are the same instance
    for(const auto& container : containers)
    {
        EXPECT_EQ(container.get(), containers[0].get());
    }
}

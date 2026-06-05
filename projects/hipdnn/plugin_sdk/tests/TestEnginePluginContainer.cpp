// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/SharedContainerManager.hpp>

using namespace hipdnn_plugin_sdk;

namespace
{

// Define test handle, settings, and execution context structs for testing
struct TestHandle
{
};

struct TestSettings
{
};

struct TestContext
{
};

// Define type alias for readability
using TestEngineManager = EngineManager<TestHandle, TestSettings, TestContext>;

// Test container that tracks initialization and meets requirements
class TestContainer
{
public:
    TestContainer()
        : _engineManager(std::make_unique<TestEngineManager>())
    {
        instanceCount++;
    }

    ~TestContainer()
    {
        instanceCount--;
    }

    static uint32_t
        copyEngineIds(int64_t* /*engineIds*/, uint32_t /*maxEngines*/, uint32_t& numEngines)
    {
        numEngines = 0;
        return 0;
    }

    TestEngineManager& getEngineManager()
    {
        return *_engineManager;
    }

    static int instanceCount;

private:
    std::unique_ptr<TestEngineManager> _engineManager;
};

int TestContainer::instanceCount = 0;

} // namespace

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

    constexpr int K_NUM_THREADS = 10;
    std::vector<std::thread> threads;
    threads.reserve(K_NUM_THREADS);

    for(int i = 0; i < K_NUM_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            auto container = manager.getOrCreate();
            const std::lock_guard<std::mutex> lock(containersMutex);
            containers.push_back(container);
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    // All threads should have gotten the same container
    EXPECT_EQ(TestContainer::instanceCount, 1);
    EXPECT_EQ(containers.size(), static_cast<size_t>(K_NUM_THREADS));

    // Verify all containers are the same instance
    for(const auto& container : containers)
    {
        EXPECT_EQ(container.get(), containers[0].get());
    }
}

TEST(TestEnginePluginContainer, ValidContainerPassesCompileTimeValidation)
{
    // Test that the compile-time validation works for containers meeting requirements
    constexpr bool K_VALID = (validateContainerType<TestContainer>(), true);
    EXPECT_TRUE(K_VALID);
}

namespace
{

// Test handle that meets all requirements for validateHandleType
struct TestValidHandle
{
    std::shared_ptr<TestContainer> container;

    void setStream(hipStream_t /*stream*/) {}

    TestEngineManager& getEngineManager()
    {
        return container->getEngineManager();
    }

    void removeEngineDetailsDetachedBuffer(const void* /*ptr*/) {}
};

} // namespace

TEST(TestEnginePluginContainer, ValidHandlePassesCompileTimeValidation)
{
    // Test that the compile-time validation works for handles meeting requirements
    constexpr bool K_VALID = (validateHandleType<TestValidHandle, TestContainer>(), true);
    EXPECT_TRUE(K_VALID);
}

TEST(TestEnginePluginContainer, HandleTypeTraitsDetectContainerMember)
{
    EXPECT_TRUE((HasContainerMember<TestValidHandle, TestContainer>::value));
}

TEST(TestEnginePluginContainer, HandleTypeTraitsDetectSetStream)
{
    EXPECT_TRUE((HasSetStream<TestValidHandle>::value));
}

TEST(TestEnginePluginContainer, HandleTypeTraitsDetectGetEngineManager)
{
    EXPECT_TRUE((HasGetEngineManager<TestValidHandle>::value));
}

TEST(TestEnginePluginContainer, HandleTypeTraitsDetectRemoveEngineDetailsDetachedBuffer)
{
    EXPECT_TRUE((HasRemoveEngineDetailsDetachedBuffer<TestValidHandle>::value));
}

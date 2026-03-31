// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "HipdnnException.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/DescriptorFactory.hpp"
#include "descriptors/DescriptorTestUtils.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/FlatbufferTestUtils.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TestMacros.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include "descriptors/mocks/MockDescriptor.hpp"
#include "plugin/EnginePluginResourceManager.hpp"
#include "plugins/mocks/MockEnginePlugin.hpp"
#include "plugins/mocks/MockEnginePluginManager.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

TEST(TestEnginePluginResourceManager, PluginLoading)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);
    }
}

TEST(TestEnginePluginResourceManager, SetStream)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mockPlugin,
                setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), hipStream_t(0x12345678)));

    EXPECT_CALL(*mockPlugin, destroyHandle(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        resourceManager.setStream(hipStream_t(0x12345678));
    }
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementSetAndGetSinglePath)
{
    std::vector<std::filesystem::path> pluginPaths = {"/test/plugin/path"};

    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    const std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementSetAndGetMultiplePaths)
{
    std::vector<std::filesystem::path> pluginPaths
        = {"/test/plugin/path1", "/test/plugin/path2", "/test/plugin/path3"};

    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    const std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementAdditiveLoadingMode)
{
    const std::vector<std::filesystem::path> initialPaths = {"/test/path1"};
    EnginePluginResourceManager::setPluginPaths(initialPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const std::vector<std::filesystem::path> additionalPaths = {"/test/path2", "/test/path3"};
    EnginePluginResourceManager::setPluginPaths(additionalPaths, HIPDNN_PLUGIN_LOADING_ADDITIVE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    const std::set<std::filesystem::path> expectedPaths
        = {"/test/path1", "/test/path2", "/test/path3"};
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementAbsoluteLoadingModeReplacesExisting)
{
    const std::vector<std::filesystem::path> initialPaths = {"/test/path1", "/test/path2"};
    EnginePluginResourceManager::setPluginPaths(initialPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> newPaths = {"/test/path3", "/test/path4"};
    EnginePluginResourceManager::setPluginPaths(newPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    const std::set<std::filesystem::path> expectedPaths(newPaths.begin(), newPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementEmptyPathsClearing)
{
    const std::vector<std::filesystem::path> pluginPaths = {"/test/path1", "/test/path2"};
    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const std::vector<std::filesystem::path> emptyPaths;
    EnginePluginResourceManager::setPluginPaths(emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();
    EXPECT_TRUE(retrievedPaths.empty());
}

TEST(TestEnginePluginResourceManager, MoveConstructor)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EnginePluginResourceManager rm1(pluginManager);

    const EnginePluginResourceManager rm2 = std::move(rm1);

    EXPECT_NO_THROW(rm2.setStream(nullptr));
}

TEST(TestEnginePluginResourceManager, MoveAssignment)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin1 = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> mockPlugin2 = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins1{mockPlugin1};
    std::vector<std::shared_ptr<EnginePlugin>> plugins2{mockPlugin2};
    const std::shared_ptr<MockEnginePluginManager> pluginManager1
        = std::make_shared<MockEnginePluginManager>();
    const std::shared_ptr<MockEnginePluginManager> pluginManager2
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager1, getPlugins()).WillOnce(::testing::ReturnRef(plugins1));
    EXPECT_CALL(*mockPlugin1, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin1, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin1, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mockPlugin1, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*pluginManager2, getPlugins()).WillOnce(::testing::ReturnRef(plugins2));
    EXPECT_CALL(*mockPlugin2, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xcafebabe)));
    EXPECT_CALL(*mockPlugin2, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{200}));
    EXPECT_CALL(*mockPlugin2, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xcafebabe))))
        .Times(testing::AtMost(1));

    EnginePluginResourceManager rm1(pluginManager1);
    EnginePluginResourceManager rm2(pluginManager2);

    rm2 = std::move(rm1);

    EXPECT_NO_THROW(rm2.setStream(nullptr));
}

TEST(TestEnginePluginResourceManager, SelfMoveAssignment)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101}));

    EXPECT_CALL(*mockPlugin, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr)).Times(2);

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));
    EXPECT_CALL(*mockPlugin,
                getApplicableEngineIds(hipdnnEnginePluginHandle_t(0xdeadbeef), testing::_))
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101}));

    std::set<std::filesystem::path> expectedPluginFiles = {"/path/to/plugin.so"};
    EXPECT_CALL(*pluginManager, getLoadedPluginFiles())
        .WillOnce(::testing::ReturnRef(expectedPluginFiles));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EnginePluginResourceManager rm(pluginManager);

    EXPECT_NO_THROW(rm.setStream(nullptr));

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    rm = std::move(rm);
#pragma clang diagnostic pop

    EXPECT_NO_THROW(rm.setStream(nullptr));

    auto engineIds = rm.getApplicableEngineIds(&mockGraphDesc);
    EXPECT_EQ(engineIds.size(), 2);
    EXPECT_EQ(engineIds[0], 100);
    EXPECT_EQ(engineIds[1], 101);

    size_t numPlugins = 0;
    size_t maxStringLen = 0;
    EXPECT_NO_THROW(rm.getLoadedPluginFiles(&numPlugins, nullptr, &maxStringLen));
    EXPECT_EQ(numPlugins, 1);
    EXPECT_GT(maxStringLen, 0);
}

TEST(TestEnginePluginResourceManager, RapidCreationDestruction)
{
    const int numIterations = 100;

    for(int i = 0; i < numIterations; ++i)
    {
        auto pluginManager = std::make_shared<MockEnginePluginManager>();
        auto mockPlugin = std::make_shared<MockEnginePlugin>();
        std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

        EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
        EXPECT_CALL(*mockPlugin, createHandle())
            .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
        EXPECT_CALL(*mockPlugin, getAllEngineIds())
            .WillOnce(::testing::Return(std::vector<int64_t>{100}));
        EXPECT_CALL(*mockPlugin,
                    destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

        {
            const EnginePluginResourceManager rm(pluginManager);
        }
    }
}

TEST(TestEnginePluginResourceManager, ConcurrentCreationAndPublicMethods)
{
    const size_t numThreads = 4;
    const size_t managersPerThread = 10;
    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<EnginePluginResourceManager>>> allManagers(numThreads);
    std::vector<std::vector<std::shared_ptr<MockEnginePluginManager>>> allPluginManagers(
        numThreads);
    std::vector<std::vector<std::shared_ptr<MockEnginePlugin>>> allMockPlugins(numThreads);
    std::vector<std::vector<std::vector<std::shared_ptr<EnginePlugin>>>> allPlugins(numThreads);
    std::atomic<size_t> successfulCreations{0};

    threads.reserve(numThreads);

    for(size_t t = 0; t < numThreads; ++t)
    {
        allManagers[t].reserve(managersPerThread);
        allPluginManagers[t].reserve(managersPerThread);
        allMockPlugins[t].reserve(managersPerThread);
        allPlugins[t].reserve(managersPerThread);
    }

    for(size_t t = 0; t < numThreads; ++t)
    {
        threads.emplace_back([t,
                              &allManagers,
                              &allPluginManagers,
                              &allMockPlugins,
                              &allPlugins,
                              &successfulCreations]() {
            for(size_t i = 0; i < managersPerThread; ++i)
            {
                auto pluginManager = std::make_shared<MockEnginePluginManager>();
                auto mockPlugin = std::make_shared<MockEnginePlugin>();
                const std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

                allPluginManagers[t].push_back(pluginManager);
                allMockPlugins[t].push_back(mockPlugin);
                allPlugins[t].push_back(plugins);

                EXPECT_CALL(*pluginManager, getPlugins())
                    .WillOnce(::testing::ReturnRef(allPlugins[t][i]));
                EXPECT_CALL(*mockPlugin, createHandle())
                    .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
                EXPECT_CALL(*mockPlugin, getAllEngineIds())
                    .WillOnce(::testing::Return(
                        std::vector<int64_t>{static_cast<int64_t>(100 + (t * 1000) + i)}));
                EXPECT_CALL(*mockPlugin,
                            setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
                EXPECT_CALL(*mockPlugin,
                            destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

                allManagers[t].push_back(
                    std::make_shared<EnginePluginResourceManager>(pluginManager));
                successfulCreations++;

                EXPECT_NO_THROW(allManagers[t].back()->setStream(nullptr));
            }
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(successfulCreations.load(), numThreads * managersPerThread);

    allManagers.clear();
}

TEST(TestEnginePluginResourceManager, GetApplicableEngineIdsNullGraphDescriptor)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.getApplicableEngineIds(nullptr),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, SetNullStream)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        EXPECT_NO_THROW(resourceManager.setStream(nullptr));
    }
}

TEST(TestEnginePluginResourceManager, GetApplicableEngineIdsWithLoadedPlugin)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));

    EXPECT_CALL(*mockPlugin,
                getApplicableEngineIds(
                    hipdnnEnginePluginHandle_t(0xdeadbeef),
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fakeSerializedData.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size, fakeSerializedData.size)))))
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);

        EXPECT_EQ(engineIds.size(), 3);
        EXPECT_EQ(engineIds[0], 100);
        EXPECT_EQ(engineIds[1], 101);
        EXPECT_EQ(engineIds[2], 102);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSize)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    const hipdnnPluginConstData_t fakeSerializedData = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));

    EXPECT_CALL(*mockPlugin,
                getWorkspaceSize(
                    hipdnnEnginePluginHandle_t(0xdeadbeef),
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fakeEngineConfig.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size, fakeEngineConfig.size))),
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fakeSerializedData.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size, fakeSerializedData.size)))))
        .WillOnce(::testing::Return(size_t(8192)));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        auto workspaceSize
            = resourceManager.getWorkspaceSize(100, &fakeEngineConfig, &mockGraphDesc);

        EXPECT_EQ(workspaceSize, 8192);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeFromExecutionContext)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, getWorkspaceSize(_, _)).WillOnce(::testing::Return(size_t(4096)));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    const EnginePluginResourceManager resourceManager(pluginManager);

    auto workspaceSize = resourceManager.getWorkspaceSize(
        100, reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x12345678));
    EXPECT_EQ(workspaceSize, 4096);
}

TEST(TestEnginePluginResourceManager, GetEngineDetails)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));

    EXPECT_CALL(
        *mockPlugin,
        getEngineDetails(hipdnnEnginePluginHandle_t(0xdeadbeef),
                         int64_t(100), // engineId
                         testing::Pointee(testing::AllOf(
                             testing::Field(&hipdnnPluginConstData_t::ptr, fakeSerializedData.ptr),
                             testing::Field(&hipdnnPluginConstData_t::size,
                                            fakeSerializedData.size))), // opGraph
                         testing::_ // output engineDetails
                         ))
        .WillOnce(testing::Invoke([](hipdnnEnginePluginHandle_t,
                                     int64_t engineId,
                                     const hipdnnPluginConstData_t*,
                                     hipdnnPluginConstData_t* output) {
            // Create valid flatbuffer engine details
            static auto s_builder = createValidEngineDetails(engineId);
            output->ptr = s_builder.GetBufferPointer();
            output->size = s_builder.GetSize();
        }));

    EXPECT_CALL(*mockPlugin,
                destroyEngineDetails(hipdnnEnginePluginHandle_t(0xdeadbeef), testing::_));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        // Test getEngineDetails functionality with valid flatbuffer data
        auto engineDetails = EnginePluginResourceManager::getEngineDetails(
            std::make_shared<EnginePluginResourceManager>(std::move(resourceManager)),
            100,
            &mockGraphDesc);

        // Verify that we got valid engine details
        EXPECT_NE(engineDetails, nullptr);
        EXPECT_NE(engineDetails->get(), nullptr);
        EXPECT_EQ(engineDetails->get()->engine_id(), 100);
    }
}

TEST(TestEnginePluginResourceManager, CreateExecutionContext)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    const hipdnnPluginConstData_t fakeSerializedData = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));

    EXPECT_CALL(*mockPlugin,
                createExecutionContext(
                    hipdnnEnginePluginHandle_t(0xdeadbeef),
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fakeEngineConfig.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size, fakeEngineConfig.size))),
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fakeSerializedData.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size, fakeSerializedData.size)))))
        .WillOnce(::testing::Return(hipdnnEnginePluginExecutionContext_t(0xcafebabe)));

    EXPECT_CALL(*mockPlugin,
                destroyExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                        hipdnnEnginePluginExecutionContext_t(0xcafebabe)));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        auto executionContext = EnginePluginResourceManager::createExecutionContext(
            std::make_shared<EnginePluginResourceManager>(std::move(resourceManager)),
            100,
            &fakeEngineConfig,
            &mockGraphDesc);

        EXPECT_NE(executionContext, nullptr);
        EXPECT_NE(executionContext->get(), nullptr);
    }
}

TEST(TestEnginePluginResourceManager, CreateExecutionContextWithInvalidEngineId)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        // Try to create execution context with an invalid engine ID (999 is not in the list)
        ASSERT_THROW_HIPDNN_STATUS(
            EnginePluginResourceManager::createExecutionContext(
                std::make_shared<EnginePluginResourceManager>(std::move(resourceManager)),
                999, // Invalid engine ID
                &fakeEngineConfig,
                &mockGraphDesc),
            HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphWithNullParameters)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.executeOpGraph(nullptr, nullptr),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailNonFinalizedPlan)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

    auto mockExecutionPlan = MockDescriptorUtility::asDescriptorUnsafe<MockExecutionPlanDescriptor>(
        executionPlanWrapper.get());
    auto mockVariantPack
        = MockDescriptorUtility::asDescriptorUnsafe<MockVariantDescriptor>(variantWrapper.get());

    const std::vector<int64_t> tensorIds = {1, 2, 3};
    const std::vector<const void*> dataPtrs = {reinterpret_cast<void*>(0x1000),
                                               reinterpret_cast<void*>(0x2000),
                                               reinterpret_cast<void*>(0x3000)};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(false));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailNonFinalizedVariant)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

    auto mockExecutionPlan = MockDescriptorUtility::asDescriptorUnsafe<MockExecutionPlanDescriptor>(
        executionPlanWrapper.get());
    auto mockVariantPack
        = MockDescriptorUtility::asDescriptorUnsafe<MockVariantDescriptor>(variantWrapper.get());

    const std::vector<int64_t> tensorIds = {1, 2, 3};
    const std::vector<const void*> dataPtrs = {reinterpret_cast<void*>(0x1000),
                                               reinterpret_cast<void*>(0x2000),
                                               reinterpret_cast<void*>(0x3000)};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(false));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailTensorMismatch)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto engineConfigWrapper = createDescriptor<MockEngineConfigDescriptor>();
    auto engineWrapper = createDescriptor<MockEngineDescriptor>();
    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

    auto mockEngineConfig = MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
        engineConfigWrapper.get());
    auto mockEngine
        = MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(engineWrapper.get());
    auto mockExecutionPlan = MockDescriptorUtility::asDescriptorUnsafe<MockExecutionPlanDescriptor>(
        executionPlanWrapper.get());
    auto mockVariantPack
        = MockDescriptorUtility::asDescriptorUnsafe<MockVariantDescriptor>(variantWrapper.get());

    // More data ptrs than tensor ids
    std::vector<int64_t> tensorIds = {1};
    std::vector<const void*> dataPtrs = {reinterpret_cast<void*>(0x1000),
                                         reinterpret_cast<void*>(0x2000),
                                         reinterpret_cast<void*>(0x3000)};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(true));

    EXPECT_CALL(*mockExecutionPlan, getEngineConfig())
        .WillOnce(::testing::Return(mockEngineConfig));
    EXPECT_CALL(*mockEngineConfig, getEngine()).WillOnce(::testing::Return(mockEngine));
    EXPECT_CALL(*mockEngine, getEngineId()).WillOnce(::testing::Return(int64_t(100)));
    EXPECT_CALL(*mockVariantPack, getWorkspace())
        .WillOnce(::testing::Return(reinterpret_cast<void*>(0x4000)));
    EXPECT_CALL(*mockVariantPack, getTensorIds()).WillOnce(::testing::ReturnRef(tensorIds));
    EXPECT_CALL(*mockVariantPack, getDataPointers()).WillOnce(::testing::ReturnRef(dataPtrs));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

namespace
{
// NOLINTNEXTLINE(readability-identifier-naming)
MATCHER_P2(MatchesMemory, data, size, "")
{
    return memcmp(arg, data, size) == 0;
}
} // namespace

TEST(TestEnginePluginResourceManager, ExecuteOpGraphSuccessWithValidDescriptors)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto engineConfigWrapper = createDescriptor<MockEngineConfigDescriptor>();
    auto engineWrapper = createDescriptor<MockEngineDescriptor>();
    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

    auto mockEngineConfig = MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
        engineConfigWrapper.get());
    auto mockEngine
        = MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(engineWrapper.get());
    auto mockExecutionPlan = MockDescriptorUtility::asDescriptorUnsafe<MockExecutionPlanDescriptor>(
        executionPlanWrapper.get());
    auto mockVariantPack
        = MockDescriptorUtility::asDescriptorUnsafe<MockVariantDescriptor>(variantWrapper.get());

    std::vector<int64_t> tensorIds = {1, 2, 3};
    std::vector<const void*> dataPtrs = {reinterpret_cast<void*>(0x1000),
                                         reinterpret_cast<void*>(0x2000),
                                         reinterpret_cast<void*>(0x3000)};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(true));

    EXPECT_CALL(*mockExecutionPlan, getEngineConfig())
        .WillOnce(::testing::Return(mockEngineConfig));
    EXPECT_CALL(*mockEngineConfig, getEngine()).WillOnce(::testing::Return(mockEngine));
    EXPECT_CALL(*mockEngine, getEngineId()).WillOnce(::testing::Return(int64_t(100)));
    EXPECT_CALL(*mockVariantPack, getWorkspace())
        .WillOnce(::testing::Return(reinterpret_cast<void*>(0x4000)));
    EXPECT_CALL(*mockVariantPack, getTensorIds()).WillOnce(::testing::ReturnRef(tensorIds));
    EXPECT_CALL(*mockVariantPack, getDataPointers()).WillOnce(::testing::ReturnRef(dataPtrs));
    EXPECT_CALL(*mockExecutionPlan, getExecutionContext())
        .WillOnce(::testing::Return(hipdnnEnginePluginExecutionContext_t(0xcafebabe)));

    std::vector<hipdnnPluginDeviceBuffer_t> expectedDeviceBuffers;
    expectedDeviceBuffers.reserve(tensorIds.size());
    for(size_t i = 0; i < tensorIds.size(); ++i)
    {
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = tensorIds[i];
        buffer.ptr = const_cast<void*>(dataPtrs[i]);
        expectedDeviceBuffers.push_back(buffer);
    }

    EXPECT_CALL(*mockPlugin,
                executeOpGraph(hipdnnEnginePluginHandle_t(0xdeadbeef),
                               hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                               reinterpret_cast<void*>(0x4000),
                               MatchesMemory(expectedDeviceBuffers.data(),
                                             expectedDeviceBuffers.size()
                                                 * sizeof(hipdnnPluginDeviceBuffer_t)),
                               static_cast<uint32_t>(tensorIds.size())));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get());
    }
}

TEST(TestEnginePluginResourceManager, GetLoadedPluginFiles)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    std::set<std::filesystem::path> expectedPluginFiles
        = {"/path/to/plugin1.so", "/path/to/plugin2.so"};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*pluginManager, getLoadedPluginFiles())
        .Times(2)
        .WillRepeatedly(::testing::ReturnRef(expectedPluginFiles));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        size_t numPlugins = 0;
        size_t maxStringLen = 0;

        EXPECT_NO_THROW(resourceManager.getLoadedPluginFiles(&numPlugins, nullptr, &maxStringLen));

        EXPECT_EQ(numPlugins, 2);
        EXPECT_GT(maxStringLen, 0);

        std::vector<std::string> pluginStrings(numPlugins);
        std::vector<char*> pluginPaths(numPlugins);
        for(size_t i = 0; i < numPlugins; ++i)
        {
            pluginStrings[i].resize(maxStringLen);
            pluginPaths[i] = pluginStrings[i].data();
        }

        EXPECT_NO_THROW(
            resourceManager.getLoadedPluginFiles(&numPlugins, pluginPaths.data(), &maxStringLen));

        // Note: std::set ordering may differ, so we check that both paths are present
        const std::set<std::string> returnedPaths
            = {std::string(pluginPaths[0]), std::string(pluginPaths[1])};
        const std::set<std::string> expectedPaths = {"/path/to/plugin1.so", "/path/to/plugin2.so"};
        EXPECT_EQ(returnedPaths, expectedPaths);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeNullEngineConfig)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.getWorkspaceSize(100, nullptr, &mockGraphDesc),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeThrowsExceptionForInvalidEngineId)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.getWorkspaceSize(200, &fakeEngineConfig, &mockGraphDesc),
            HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeFromExecutionContextNullExecutionContext)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    const EnginePluginResourceManager resourceManager(pluginManager);

    ASSERT_THROW_HIPDNN_STATUS(resourceManager.getWorkspaceSize(100, nullptr),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestEnginePluginResourceManager,
     GetWorkspaceSizeFromExecutionContextThrowsExceptionForInvalidEngineId)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    const EnginePluginResourceManager resourceManager(pluginManager);

    ASSERT_THROW_HIPDNN_STATUS(
        resourceManager.getWorkspaceSize(
            200, reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x12345678)),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestEnginePluginResourceManager, SetPluginPathsWithActiveResourceManager)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);
        std::vector<std::filesystem::path> pluginPaths = {"/test/path"};

        EXPECT_NO_THROW(EnginePluginResourceManager::setPluginPaths(
            pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

        auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();
        const std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
        EXPECT_EQ(retrievedPaths, expectedPaths);

        const std::vector<std::filesystem::path> emptyPaths;
        EXPECT_NO_THROW(EnginePluginResourceManager::setPluginPaths(
            emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    }
}

TEST(TestEnginePluginResourceManager, ConstructorSkipsPluginWhenCreateHandleThrowsHipdnnException)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, name()).WillRepeatedly(::testing::Return("BadPlugin"));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Throw(HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Test error")));

    // Plugin should never be queried since it failed to load
    EXPECT_CALL(*mockPlugin, getApplicableEngineIds(testing::_, testing::_)).Times(0);

    // No destroyHandle call expected since handle creation failed

    {
        // Constructor should not throw, but the plugin should be skipped
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify no engines were registered
        EXPECT_CALL(mockGraphDesc, getSerializedGraph())
            .WillOnce(::testing::Return(fakeSerializedData));
        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_TRUE(engineIds.empty());
    }
}

TEST(TestEnginePluginResourceManager, ConstructorSkipsPluginWhenCreateHandleThrowsStdException)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, name()).WillRepeatedly(::testing::Return("BadPlugin"));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Throw(std::runtime_error("Test std::exception")));

    // Plugin should never be queried since it failed to load
    EXPECT_CALL(*mockPlugin, getApplicableEngineIds(testing::_, testing::_)).Times(0);

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify no engines were registered
        EXPECT_CALL(mockGraphDesc, getSerializedGraph())
            .WillOnce(::testing::Return(fakeSerializedData));
        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_TRUE(engineIds.empty());
    }
}

TEST(TestEnginePluginResourceManager, ConstructorSkipsPluginWhenCreateHandleReturnsNull)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, name()).WillRepeatedly(::testing::Return("NullHandlePlugin"));
    EXPECT_CALL(*mockPlugin, createHandle()).WillOnce(::testing::Return(nullptr));

    // Plugin should never be queried since it returned null handle
    EXPECT_CALL(*mockPlugin, getApplicableEngineIds(testing::_, testing::_)).Times(0);

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify no engines were registered
        EXPECT_CALL(mockGraphDesc, getSerializedGraph())
            .WillOnce(::testing::Return(fakeSerializedData));
        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_TRUE(engineIds.empty());
    }
}

TEST(TestEnginePluginResourceManager, ConstructorSkipsPluginOnHandleCollision)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin1 = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> mockPlugin2 = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin1, mockPlugin2};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    // Both plugins return the same handle (simulating a collision)
    auto collisionHandle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x123);

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    // First plugin succeeds
    EXPECT_CALL(*mockPlugin1, createHandle()).WillOnce(::testing::Return(collisionHandle));
    EXPECT_CALL(*mockPlugin1, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin1, destroyHandle(collisionHandle));

    // Second plugin returns same handle - should be skipped
    EXPECT_CALL(*mockPlugin2, name()).WillRepeatedly(::testing::Return("CollidingPlugin"));
    EXPECT_CALL(*mockPlugin2, createHandle()).WillOnce(::testing::Return(collisionHandle));
    EXPECT_CALL(*mockPlugin2, destroyHandle(collisionHandle));

    // Second plugin should never be queried since it had handle collision
    EXPECT_CALL(*mockPlugin2, getApplicableEngineIds(testing::_, testing::_)).Times(0);

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify first plugin loaded successfully, second was skipped
        EXPECT_CALL(mockGraphDesc, getSerializedGraph())
            .WillOnce(::testing::Return(fakeSerializedData));
        EXPECT_CALL(*mockPlugin1, getApplicableEngineIds(collisionHandle, testing::_))
            .WillOnce(::testing::Return(std::vector<int64_t>{100}));

        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_EQ(engineIds.size(), 1);
        EXPECT_EQ(engineIds[0], 100);
    }
}

TEST(TestEnginePluginResourceManager, ConstructorSkipsPluginWhenGetAllEngineIdsThrows)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef);

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, name()).WillRepeatedly(::testing::Return("EngineIdFailPlugin"));
    EXPECT_CALL(*mockPlugin, createHandle()).WillOnce(::testing::Return(handle));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Throw(
            HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Failed to get engine IDs")));
    EXPECT_CALL(*mockPlugin, destroyHandle(handle));

    // Plugin should never be queried since getAllEngineIds failed
    EXPECT_CALL(*mockPlugin, getApplicableEngineIds(testing::_, testing::_)).Times(0);

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify plugin was skipped and handle was cleaned up
        EXPECT_CALL(mockGraphDesc, getSerializedGraph())
            .WillOnce(::testing::Return(fakeSerializedData));
        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_TRUE(engineIds.empty());
    }
}

TEST(TestEnginePluginResourceManager, ConstructorContinuesAfterBadPluginWithGoodPlugin)
{
    const std::shared_ptr<MockEnginePlugin> badPlugin = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> goodPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{badPlugin, goodPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    // Bad plugin throws during createHandle
    EXPECT_CALL(*badPlugin, name()).WillRepeatedly(::testing::Return("BadPlugin"));
    EXPECT_CALL(*badPlugin, createHandle())
        .WillOnce(::testing::Throw(HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Test error")));

    // Good plugin succeeds
    auto goodHandle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0xcafe);
    EXPECT_CALL(*goodPlugin, createHandle()).WillOnce(::testing::Return(goodHandle));
    EXPECT_CALL(*goodPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{200, 201}));
    EXPECT_CALL(*goodPlugin, destroyHandle(goodHandle));

    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));
    EXPECT_CALL(*goodPlugin, getApplicableEngineIds(goodHandle, testing::_))
        .WillOnce(::testing::Return(std::vector<int64_t>{200, 201}));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        // Verify good plugin's engines are available
        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
        EXPECT_EQ(engineIds.size(), 2);
        EXPECT_EQ(engineIds[0], 200);
        EXPECT_EQ(engineIds[1], 201);
    }
}

TEST(TestEnginePluginResourceManager, GetEngineInfosSinglePlugin)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillRepeatedly(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillRepeatedly(::testing::Return(std::vector<int64_t>{100, 101}));
    EXPECT_CALL(*mockPlugin, name()).WillRepeatedly(::testing::Return("test-plugin"));
    EXPECT_CALL(*mockPlugin, version()).WillRepeatedly(::testing::Return("1.0"));
    EXPECT_CALL(*mockPlugin, type()).WillRepeatedly(::testing::Return(HIPDNN_PLUGIN_TYPE_ENGINE));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        auto infos = resourceManager.getEngineInfos();

        ASSERT_EQ(infos.size(), 2);

        // Results are sorted by engineName. formatEngineIdHex(100) = "0x0000000000000064",
        // formatEngineIdHex(101) = "0x0000000000000065"
        EXPECT_EQ(infos[0].engineId, 100);
        EXPECT_EQ(infos[0].engineName, "0x0000000000000064");
        EXPECT_EQ(infos[0].pluginName, "test-plugin");
        EXPECT_EQ(infos[0].version, "1.0");
        EXPECT_EQ(infos[0].type, "HIPDNN_PLUGIN_TYPE_ENGINE");

        EXPECT_EQ(infos[1].engineId, 101);
        EXPECT_EQ(infos[1].engineName, "0x0000000000000065");
        EXPECT_EQ(infos[1].pluginName, "test-plugin");
        EXPECT_EQ(infos[1].version, "1.0");
        EXPECT_EQ(infos[1].type, "HIPDNN_PLUGIN_TYPE_ENGINE");
    }
}

TEST(TestEnginePluginResourceManager, GetEngineInfosMultiplePlugins)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin1 = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> mockPlugin2 = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin1, mockPlugin2};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillRepeatedly(::testing::ReturnRef(plugins));

    EXPECT_CALL(*mockPlugin1, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin1, getAllEngineIds())
        .WillRepeatedly(::testing::Return(std::vector<int64_t>{200}));
    EXPECT_CALL(*mockPlugin1, name()).WillRepeatedly(::testing::Return("plugin-alpha"));
    EXPECT_CALL(*mockPlugin1, version()).WillRepeatedly(::testing::Return("2.0"));
    EXPECT_CALL(*mockPlugin1, type()).WillRepeatedly(::testing::Return(HIPDNN_PLUGIN_TYPE_ENGINE));
    EXPECT_CALL(*mockPlugin1, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*mockPlugin2, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xcafebabe)));
    EXPECT_CALL(*mockPlugin2, getAllEngineIds())
        .WillRepeatedly(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin2, name()).WillRepeatedly(::testing::Return("plugin-beta"));
    EXPECT_CALL(*mockPlugin2, version()).WillRepeatedly(::testing::Return("3.0"));
    EXPECT_CALL(*mockPlugin2, type())
        .WillRepeatedly(::testing::Return(HIPDNN_PLUGIN_TYPE_UNSPECIFIED));
    EXPECT_CALL(*mockPlugin2, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xcafebabe))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        auto infos = resourceManager.getEngineInfos();

        ASSERT_EQ(infos.size(), 2);

        // Sorted by engineName: "0x0000000000000064" (100) < "0x00000000000000C8" (200)
        EXPECT_EQ(infos[0].engineId, 100);
        EXPECT_EQ(infos[0].engineName, "0x0000000000000064");
        EXPECT_EQ(infos[0].pluginName, "plugin-beta");
        EXPECT_EQ(infos[0].version, "3.0");
        EXPECT_EQ(infos[0].type, "HIPDNN_PLUGIN_TYPE_UNSPECIFIED");

        EXPECT_EQ(infos[1].engineId, 200);
        EXPECT_EQ(infos[1].engineName, "0x00000000000000C8");
        EXPECT_EQ(infos[1].pluginName, "plugin-alpha");
        EXPECT_EQ(infos[1].version, "2.0");
        EXPECT_EQ(infos[1].type, "HIPDNN_PLUGIN_TYPE_ENGINE");
    }
}

TEST(TestEnginePluginResourceManager, GetEngineInfosNoPlugins)
{
    std::vector<std::shared_ptr<EnginePlugin>> plugins;
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillRepeatedly(::testing::ReturnRef(plugins));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        auto infos = resourceManager.getEngineInfos();

        EXPECT_TRUE(infos.empty());
    }
}

// Test subclass to access the protected default constructor
class TestableEnginePluginResourceManager : public EnginePluginResourceManager
{
public:
    TestableEnginePluginResourceManager() = default;
};

TEST(TestEnginePluginResourceManager, GetEngineInfosNullPluginManager)
{
    const TestableEnginePluginResourceManager resourceManager;

    auto infos = resourceManager.getEngineInfos();

    EXPECT_TRUE(infos.empty());
}

TEST(TestEnginePluginResourceManager, ConstructorHandlesMultipleBadPlugins)
{
    const std::shared_ptr<MockEnginePlugin> nullPlugin = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> throwingPlugin = std::make_shared<MockEnginePlugin>();
    const std::shared_ptr<MockEnginePlugin> goodPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{nullPlugin, throwingPlugin, goodPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    // First plugin returns null
    EXPECT_CALL(*nullPlugin, name()).WillRepeatedly(::testing::Return("NullPlugin"));
    EXPECT_CALL(*nullPlugin, createHandle()).WillOnce(::testing::Return(nullptr));

    // Second plugin throws
    EXPECT_CALL(*throwingPlugin, name()).WillRepeatedly(::testing::Return("ThrowingPlugin"));
    EXPECT_CALL(*throwingPlugin, createHandle())
        .WillOnce(::testing::Throw(std::runtime_error("Plugin initialization failed")));

    // Third plugin succeeds
    auto goodHandle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0xbeef);
    EXPECT_CALL(*goodPlugin, createHandle()).WillOnce(::testing::Return(goodHandle));
    EXPECT_CALL(*goodPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{300}));
    EXPECT_CALL(*goodPlugin, setStream(goodHandle, nullptr));
    EXPECT_CALL(*goodPlugin, destroyHandle(goodHandle));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);
        // Verify good plugin was loaded by calling setStream
        EXPECT_NO_THROW(resourceManager.setStream(nullptr));
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

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
using namespace ::testing;

TEST(TestEnginePluginResourceManager, PluginLoading)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    {
        EnginePluginResourceManager resourceManager(pluginManager);
    }
}

TEST(TestEnginePluginResourceManager, SetStream)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

    std::shared_ptr<MockEnginePluginManager> pluginManager
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
        EnginePluginResourceManager resourceManager(pluginManager);

        resourceManager.setStream(hipStream_t(0x12345678));
    }
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementSetAndGetSinglePath)
{
    std::vector<std::filesystem::path> pluginPaths = {"/test/plugin/path"};

    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementSetAndGetMultiplePaths)
{
    std::vector<std::filesystem::path> pluginPaths
        = {"/test/plugin/path1", "/test/plugin/path2", "/test/plugin/path3"};

    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementAdditiveLoadingMode)
{
    std::vector<std::filesystem::path> initialPaths = {"/test/path1"};
    EnginePluginResourceManager::setPluginPaths(initialPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> additionalPaths = {"/test/path2", "/test/path3"};
    EnginePluginResourceManager::setPluginPaths(additionalPaths, HIPDNN_PLUGIN_LOADING_ADDITIVE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    std::set<std::filesystem::path> expectedPaths = {"/test/path1", "/test/path2", "/test/path3"};
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementAbsoluteLoadingModeReplacesExisting)
{
    std::vector<std::filesystem::path> initialPaths = {"/test/path1", "/test/path2"};
    EnginePluginResourceManager::setPluginPaths(initialPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> newPaths = {"/test/path3", "/test/path4"};
    EnginePluginResourceManager::setPluginPaths(newPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();

    std::set<std::filesystem::path> expectedPaths(newPaths.begin(), newPaths.end());
    EXPECT_EQ(retrievedPaths, expectedPaths);
}

TEST(TestEnginePluginResourceManager, StaticPluginPathManagementEmptyPathsClearing)
{
    std::vector<std::filesystem::path> pluginPaths = {"/test/path1", "/test/path2"};
    EnginePluginResourceManager::setPluginPaths(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> emptyPaths;
    EnginePluginResourceManager::setPluginPaths(emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();
    EXPECT_TRUE(retrievedPaths.empty());
}

TEST(TestEnginePluginResourceManager, MoveConstructor)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EnginePluginResourceManager rm1(pluginManager);

    EnginePluginResourceManager rm2 = std::move(rm1);

    EXPECT_NO_THROW(rm2.setStream(nullptr));
}

TEST(TestEnginePluginResourceManager, MoveAssignment)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin1 = std::make_shared<MockEnginePlugin>();
    std::shared_ptr<MockEnginePlugin> mockPlugin2 = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins1{mockPlugin1};
    std::vector<std::shared_ptr<EnginePlugin>> plugins2{mockPlugin2};
    std::shared_ptr<MockEnginePluginManager> pluginManager1
        = std::make_shared<MockEnginePluginManager>();
    std::shared_ptr<MockEnginePluginManager> pluginManager2
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
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeSerializedData
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
            EnginePluginResourceManager rm(pluginManager);
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
                std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};

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
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.getApplicableEngineIds(nullptr),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, SetNullStream)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, setStream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        EXPECT_NO_THROW(resourceManager.setStream(nullptr));
    }
}

TEST(TestEnginePluginResourceManager, GetApplicableEngineIdsWithLoadedPlugin)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeSerializedData = {
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
        EnginePluginResourceManager resourceManager(pluginManager);

        auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);

        EXPECT_EQ(engineIds.size(), 3);
        EXPECT_EQ(engineIds[0], 100);
        EXPECT_EQ(engineIds[1], 101);
        EXPECT_EQ(engineIds[2], 102);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSize)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    hipdnnPluginConstData_t fakeSerializedData = {
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
        EnginePluginResourceManager resourceManager(pluginManager);

        auto workspaceSize
            = resourceManager.getWorkspaceSize(100, &fakeEngineConfig, &mockGraphDesc);

        EXPECT_EQ(workspaceSize, 8192);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeFromExecutionContext)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, getWorkspaceSize(_, _)).WillOnce(::testing::Return(size_t(4096)));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    EnginePluginResourceManager resourceManager(pluginManager);

    auto workspaceSize = resourceManager.getWorkspaceSize(
        100, reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x12345678));
    EXPECT_EQ(workspaceSize, 4096);
}

TEST(TestEnginePluginResourceManager, GetEngineDetails)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeSerializedData = {
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
            static auto s_builder = hipdnn_sdk::test_utilities::createValidEngineDetails(engineId);
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
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeEngineConfig = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    hipdnnPluginConstData_t fakeSerializedData = {
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
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeEngineConfig = {
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
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.executeOpGraph(nullptr, nullptr),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailNonFinalizedPlan)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto executionPlanWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockVariantDescriptor>();

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

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(false));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailNonFinalizedVariant)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto executionPlanWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockVariantDescriptor>();

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
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(false));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphFailTensorMismatch)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto engineConfigWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockEngineConfigDescriptor>();
    auto engineWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockEngineDescriptor>();
    auto executionPlanWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockVariantDescriptor>();

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
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get()),
            HIPDNN_STATUS_BAD_PARAM);
    }
}

// NOLINTNEXTLINE(readability-identifier-naming)
MATCHER_P2(MatchesMemory, data, size, "")
{
    return memcmp(arg, data, size) == 0;
}

TEST(TestEnginePluginResourceManager, ExecuteOpGraphSuccessWithValidDescriptors)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    auto engineConfigWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockEngineConfigDescriptor>();
    auto engineWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockEngineDescriptor>();
    auto executionPlanWrapper
        = hipdnn_sdk::test_utilities::createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockVariantDescriptor>();

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
        EnginePluginResourceManager resourceManager(pluginManager);

        resourceManager.executeOpGraph(executionPlanWrapper.get(), variantWrapper.get());
    }
}

TEST(TestEnginePluginResourceManager, GetLoadedPluginFiles)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
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
        EnginePluginResourceManager resourceManager(pluginManager);

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
        std::set<std::string> returnedPaths
            = {std::string(pluginPaths[0]), std::string(pluginPaths[1])};
        std::set<std::string> expectedPaths = {"/path/to/plugin1.so", "/path/to/plugin2.so"};
        EXPECT_EQ(returnedPaths, expectedPaths);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeNullEngineConfig)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(resourceManager.getWorkspaceSize(100, nullptr, &mockGraphDesc),
                                   HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeThrowsExceptionForInvalidEngineId)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    MockGraphDescriptor mockGraphDesc;
    hipdnnPluginConstData_t fakeEngineConfig = {
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

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.getWorkspaceSize(200, &fakeEngineConfig, &mockGraphDesc),
            HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, GetWorkspaceSizeFromExecutionContextNullExecutionContext)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    EnginePluginResourceManager resourceManager(pluginManager);

    ASSERT_THROW_HIPDNN_STATUS(resourceManager.getWorkspaceSize(100, nullptr),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestEnginePluginResourceManager,
     GetWorkspaceSizeFromExecutionContextThrowsExceptionForInvalidEngineId)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(reinterpret_cast<hipdnnEnginePluginHandle_t>(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(_));

    EnginePluginResourceManager resourceManager(pluginManager);

    ASSERT_THROW_HIPDNN_STATUS(
        resourceManager.getWorkspaceSize(
            200, reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x12345678)),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestEnginePluginResourceManager, SetPluginPathsWithActiveResourceManager)
{
    std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);
        std::vector<std::filesystem::path> pluginPaths = {"/test/path"};

        EXPECT_NO_THROW(EnginePluginResourceManager::setPluginPaths(
            pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

        auto retrievedPaths = EnginePluginResourceManager::getPluginPaths();
        std::set<std::filesystem::path> expectedPaths(pluginPaths.begin(), pluginPaths.end());
        EXPECT_EQ(retrievedPaths, expectedPaths);

        std::vector<std::filesystem::path> emptyPaths;
        EXPECT_NO_THROW(EnginePluginResourceManager::setPluginPaths(
            emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    }
}

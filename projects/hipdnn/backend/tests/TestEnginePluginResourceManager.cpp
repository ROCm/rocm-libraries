// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <atomic>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
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
#include "descriptors/mocks/MockEnginePluginResourceManager.hpp"
#include "plugin/EnginePluginResourceManager.hpp"
#include "plugins/mocks/MockEnginePlugin.hpp"
#include "plugins/mocks/MockEnginePluginManager.hpp"
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

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

TEST(TestEngineDetailsWrapper, DestroysPluginDetailsWhenFlatbufferVerificationFails)
{
    auto resourceManager = std::make_shared<MockEnginePluginResourceManager>();
    std::array<uint8_t, 4> malformedBytes{1, 2, 3, 4};
    hipdnnPluginConstData_t returnedDetails{malformedBytes.data(), malformedBytes.size()};

    EXPECT_CALL(*resourceManager, getEngineDetails(100, nullptr, _))
        .WillOnce([&returnedDetails](
                      int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* engineDetails) {
            *engineDetails = returnedDetails;
        });
    EXPECT_CALL(*resourceManager, destroyEngineDetails(100, _))
        .WillOnce([](int64_t, hipdnnPluginConstData_t* engineDetails) {
            EXPECT_NE(engineDetails, nullptr);
            EXPECT_NE(engineDetails->ptr, nullptr);
            EXPECT_EQ(engineDetails->size, 4u);
        });

    ASSERT_THROW_HIPDNN_STATUS(EngineDetailsWrapper(resourceManager, 100, nullptr),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestEngineDetailsWrapper, MoveAssignmentDestroysDestinationDetailsAndTransfersSourceDetails)
{
    auto resourceManager = std::make_shared<MockEnginePluginResourceManager>();
    auto detailsA = createValidEngineDetails(100);
    auto detailsB = createValidEngineDetails(200);
    hipdnnPluginConstData_t returnedDetailsA{detailsA.GetBufferPointer(), detailsA.GetSize()};
    hipdnnPluginConstData_t returnedDetailsB{detailsB.GetBufferPointer(), detailsB.GetSize()};

    EXPECT_CALL(*resourceManager, getEngineDetails(100, nullptr, _))
        .WillOnce([&returnedDetailsA](
                      int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* engineDetails) {
            *engineDetails = returnedDetailsA;
        });
    EXPECT_CALL(*resourceManager, getEngineDetails(200, nullptr, _))
        .WillOnce([&returnedDetailsB](
                      int64_t, const GraphDescriptor*, hipdnnPluginConstData_t* engineDetails) {
            *engineDetails = returnedDetailsB;
        });

    bool destroyedA = false;
    bool destroyedB = false;

    {
        EngineDetailsWrapper destination(resourceManager, 100, nullptr);
        EngineDetailsWrapper source(resourceManager, 200, nullptr);

        EXPECT_CALL(*resourceManager, destroyEngineDetails(100, _))
            .WillOnce(
                [&destroyedA, &returnedDetailsA](int64_t, hipdnnPluginConstData_t* engineDetails) {
                    destroyedA = true;
                    EXPECT_EQ(engineDetails->ptr, returnedDetailsA.ptr);
                    EXPECT_EQ(engineDetails->size, returnedDetailsA.size);
                });
        EXPECT_CALL(*resourceManager, destroyEngineDetails(200, _)).Times(0);

        destination = std::move(source);

        EXPECT_TRUE(destroyedA);
        Mock::VerifyAndClearExpectations(resourceManager.get());

        EXPECT_CALL(*resourceManager, destroyEngineDetails(200, _))
            .WillOnce(
                [&destroyedB, &returnedDetailsB](int64_t, hipdnnPluginConstData_t* engineDetails) {
                    destroyedB = true;
                    EXPECT_EQ(engineDetails->ptr, returnedDetailsB.ptr);
                    EXPECT_EQ(engineDetails->size, returnedDetailsB.size);
                });
    }

    EXPECT_TRUE(destroyedB);
}

TEST(TestEngineExecutionContextWrapper,
     MoveAssignmentDestroysDestinationContextAndTransfersSourceContext)
{
    auto resourceManager = std::make_shared<MockEnginePluginResourceManager>();
    auto contextA = hipdnnEnginePluginExecutionContext_t(0xaaaaaaaa);
    auto contextB = hipdnnEnginePluginExecutionContext_t(0xbbbbbbbb);
    const hipdnnPluginConstData_t fakeEngineConfig{reinterpret_cast<const void*>("fake_config"),
                                                   11};

    EXPECT_CALL(*resourceManager, createExecutionContext(100, &fakeEngineConfig, nullptr))
        .WillOnce(::testing::Return(contextA));
    EXPECT_CALL(*resourceManager, createExecutionContext(200, &fakeEngineConfig, nullptr))
        .WillOnce(::testing::Return(contextB));

    bool destroyedA = false;
    bool destroyedB = false;

    {
        EngineExecutionContextWrapper destination(resourceManager, 100, &fakeEngineConfig, nullptr);
        EngineExecutionContextWrapper source(resourceManager, 200, &fakeEngineConfig, nullptr);

        EXPECT_CALL(*resourceManager, destroyExecutionContext(100, contextA))
            .WillOnce([&destroyedA](int64_t, hipdnnEnginePluginExecutionContext_t) {
                destroyedA = true;
            });
        EXPECT_CALL(*resourceManager, destroyExecutionContext(200, contextB)).Times(0);

        destination = std::move(source);

        EXPECT_TRUE(destroyedA);
        Mock::VerifyAndClearExpectations(resourceManager.get());

        EXPECT_CALL(*resourceManager, destroyExecutionContext(200, contextB))
            .WillOnce([&destroyedB](int64_t, hipdnnEnginePluginExecutionContext_t) {
                destroyedB = true;
            });
    }

    EXPECT_TRUE(destroyedB);
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
    EXPECT_CALL(*mockPlugin, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE));

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
    EXPECT_CALL(*mockPlugin, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE));

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

TEST(TestEnginePluginResourceManager, SerializeExecutionContextFailsForUnsupportedPlugin)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    std::vector<uint8_t> serializedContext;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                serializeExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                          hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                          _))
        .WillOnce(::testing::Throw(
            HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                            "Engine plugin does not support execution context serialization")));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.serializeExecutionContext(
                100, hipdnnEnginePluginExecutionContext_t(0xcafebabe), serializedContext),
            HIPDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST(TestEnginePluginResourceManager, SerializeExecutionContextRejectsNullPluginPayload)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    std::vector<uint8_t> serializedContext;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                serializeExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                          hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                          _))
        .WillOnce([](hipdnnEnginePluginHandle_t,
                     hipdnnEnginePluginExecutionContext_t,
                     hipdnnPluginConstData_t* serializedContext) {
            *serializedContext = hipdnnPluginConstData_t{nullptr, 4};
        });
    EXPECT_CALL(*mockPlugin,
                destroySerializedExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef), _));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.serializeExecutionContext(
                100, hipdnnEnginePluginExecutionContext_t(0xcafebabe), serializedContext),
            HIPDNN_STATUS_PLUGIN_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, SerializeExecutionContextRejectsEmptyPluginPayload)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    const std::array<uint8_t, 1> payloadBytes{9};
    std::vector<uint8_t> serializedContext;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                serializeExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                          hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                          _))
        .WillOnce([&payloadBytes](hipdnnEnginePluginHandle_t,
                                  hipdnnEnginePluginExecutionContext_t,
                                  hipdnnPluginConstData_t* serializedContext) {
            *serializedContext = hipdnnPluginConstData_t{payloadBytes.data(), 0};
        });
    EXPECT_CALL(*mockPlugin,
                destroySerializedExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef), _));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.serializeExecutionContext(
                100, hipdnnEnginePluginExecutionContext_t(0xcafebabe), serializedContext),
            HIPDNN_STATUS_PLUGIN_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, SerializeExecutionContextCopiesPluginPayload)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    const std::array<uint8_t, 4> payloadBytes{9, 8, 7, 6};
    std::vector<uint8_t> serializedContext;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                serializeExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                          hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                          _))
        .WillOnce([&payloadBytes](hipdnnEnginePluginHandle_t,
                                  hipdnnEnginePluginExecutionContext_t,
                                  hipdnnPluginConstData_t* serializedContext) {
            *serializedContext = hipdnnPluginConstData_t{payloadBytes.data(), payloadBytes.size()};
        });
    EXPECT_CALL(*mockPlugin,
                destroySerializedExecutionContext(hipdnnEnginePluginHandle_t(0xdeadbeef), _));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_NO_THROW(resourceManager.serializeExecutionContext(
            100, hipdnnEnginePluginExecutionContext_t(0xcafebabe), serializedContext));
    }

    ASSERT_EQ(serializedContext, std::vector<uint8_t>(payloadBytes.begin(), payloadBytes.end()));
}

TEST(TestEnginePluginResourceManager, SerializeExecutionContextRejectsInvalidInputs)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    std::vector<uint8_t> serializedContext;

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        const EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.serializeExecutionContext(100, nullptr, serializedContext),
            HIPDNN_STATUS_BAD_PARAM);
        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.serializeExecutionContext(
                101, hipdnnEnginePluginExecutionContext_t(0xcafebabe), serializedContext),
            HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestEnginePluginResourceManager, CreateExecutionContextFromSerializedFailsForUnsupportedPlugin)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    const std::array<uint8_t, 3> serializedContextBytes{4, 5, 6};
    const hipdnnPluginConstData_t serializedContext{serializedContextBytes.data(),
                                                    serializedContextBytes.size()};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                createExecutionContextFromSerialized(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                                     &serializedContext))
        .WillOnce(::testing::Throw(
            HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                            "Engine plugin does not support execution context serialization")));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            EnginePluginResourceManager::createExecutionContextFromSerialized(
                std::make_shared<EnginePluginResourceManager>(std::move(resourceManager)),
                100,
                &serializedContext),
            HIPDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST(TestEnginePluginResourceManager, CreateExecutionContextFromSerializedPropagatesPluginFailure)
{
    const std::shared_ptr<MockEnginePlugin> mockPlugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{mockPlugin};
    const std::shared_ptr<MockEnginePluginManager> pluginManager
        = std::make_shared<MockEnginePluginManager>();

    const std::array<uint8_t, 3> serializedContextBytes{4, 5, 6};
    const hipdnnPluginConstData_t serializedContext{serializedContextBytes.data(),
                                                    serializedContextBytes.size()};

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mockPlugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mockPlugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mockPlugin,
                createExecutionContextFromSerialized(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                                     &serializedContext))
        .WillOnce(::testing::Throw(HipdnnException(
            HIPDNN_STATUS_PLUGIN_ERROR, "Plugin rejected serialized execution context")));
    EXPECT_CALL(*mockPlugin, destroyHandle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        EnginePluginResourceManager resourceManager(pluginManager);

        ASSERT_THROW_HIPDNN_STATUS(
            EnginePluginResourceManager::createExecutionContextFromSerialized(
                std::make_shared<EnginePluginResourceManager>(std::move(resourceManager)),
                100,
                &serializedContext),
            HIPDNN_STATUS_PLUGIN_ERROR);
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

    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

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

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(true));

    EXPECT_CALL(*mockExecutionPlan, getEngineId()).WillOnce(::testing::Return(int64_t(100)));
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

    auto executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    auto variantWrapper = createDescriptor<MockVariantDescriptor>();

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

    EXPECT_CALL(*mockExecutionPlan, isFinalized()).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockVariantPack, isFinalized()).WillOnce(::testing::Return(true));

    EXPECT_CALL(*mockExecutionPlan, getEngineId()).WillOnce(::testing::Return(int64_t(100)));
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
    EXPECT_CALL(*mockPlugin1, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE));
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
    EXPECT_CALL(*goodPlugin, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE));
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

// =============================================================================
// Override execute applicability and dispatch tests.
// =============================================================================

namespace
{

/// Programs `mockGraphDesc.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT, ...)`
/// to report `flag`. Uses the shared `SetArg4ToBool` action helper from
/// `MockDescriptor.hpp`. Other attribute lookups remain unhandled (StrictMock
/// would fail; NaggyMock will warn, which is the existing test convention).
void programOverrideFlag(MockGraphDescriptor& mockGraphDesc, bool flag)
{
    EXPECT_CALL(mockGraphDesc,
                getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                             HIPDNN_TYPE_BOOLEAN,
                             1,
                             ::testing::_,
                             ::testing::_))
        .WillRepeatedly(SetArg4ToBool(flag));
}

enum class OverrideFlagReadMode
{
    RETURNS_TRUE,
    RETURNS_FALSE,
    THROWS_NOT_SUPPORTED,
    RETURNS_COUNT_ZERO,
    THROWS_INTERNAL_ERROR
};

void programOverrideFlagRead(MockGraphDescriptor& mockGraphDesc, OverrideFlagReadMode mode)
{
    switch(mode)
    {
    case OverrideFlagReadMode::RETURNS_TRUE:
        programOverrideFlag(mockGraphDesc, /*flag=*/true);
        break;
    case OverrideFlagReadMode::RETURNS_FALSE:
        programOverrideFlag(mockGraphDesc, /*flag=*/false);
        break;
    case OverrideFlagReadMode::THROWS_NOT_SUPPORTED:
        EXPECT_CALL(mockGraphDesc,
                    getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                 HIPDNN_TYPE_BOOLEAN,
                                 1,
                                 ::testing::_,
                                 ::testing::_))
            .WillOnce(::testing::Throw(
                HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED, "override flag not supported")));
        break;
    case OverrideFlagReadMode::RETURNS_COUNT_ZERO:
        EXPECT_CALL(mockGraphDesc,
                    getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                 HIPDNN_TYPE_BOOLEAN,
                                 1,
                                 ::testing::_,
                                 ::testing::_))
            .WillOnce([](hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t,
                         int64_t,
                         int64_t* elementCount,
                         void*) { *elementCount = 0; });
        break;
    case OverrideFlagReadMode::THROWS_INTERNAL_ERROR:
        EXPECT_CALL(mockGraphDesc,
                    getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                 HIPDNN_TYPE_BOOLEAN,
                                 1,
                                 ::testing::_,
                                 ::testing::_))
            .WillOnce(::testing::Throw(
                HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "override flag read failed")));
        break;
    default:
        break;
    }
}

} // namespace

struct ApplicabilityFilterCase
{
    const char* name;
    std::string_view apiVersion;
    OverrideFlagReadMode flagReadMode;
    int hasOverrideExecute = -1; // -1: not queried, 0: queried false, 1: queried true.
    std::vector<int64_t> expectedEngineIds;
    hipdnnStatus_t expectedThrow = HIPDNN_STATUS_SUCCESS;
};

class TestEnginePluginResourceManagerApplicabilityFilter
    : public ::testing::TestWithParam<ApplicabilityFilterCase>
{
};

TEST_P(TestEnginePluginResourceManagerApplicabilityFilter, SinglePluginVersionAndFlagMatrix)
{
    const auto& testCase = GetParam();
    auto plugin = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{plugin};
    auto pluginManager = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*plugin, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*plugin, getAllEngineIds()).WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*plugin, name()).WillRepeatedly(::testing::Return("MockPlugin"));
    EXPECT_CALL(*plugin, apiVersion()).WillRepeatedly(::testing::Return(testCase.apiVersion));
    EXPECT_CALL(*plugin, destroyHandle(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};
    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));
    programOverrideFlagRead(mockGraphDesc, testCase.flagReadMode);

    if(testCase.hasOverrideExecute >= 0)
    {
        EXPECT_CALL(*plugin, hasOverrideExecute())
            .WillOnce(::testing::Return(testCase.hasOverrideExecute == 1));
    }
    else
    {
        EXPECT_CALL(*plugin, hasOverrideExecute()).Times(0);
    }

    if(testCase.expectedThrow == HIPDNN_STATUS_SUCCESS && !testCase.expectedEngineIds.empty())
    {
        EXPECT_CALL(*plugin, getApplicableEngineIds(hipdnnEnginePluginHandle_t(0xdeadbeef), _))
            .WillOnce(::testing::Return(testCase.expectedEngineIds));
    }
    else
    {
        EXPECT_CALL(*plugin, getApplicableEngineIds(_, _)).Times(0);
    }

    const EnginePluginResourceManager resourceManager(pluginManager);
    if(testCase.expectedThrow != HIPDNN_STATUS_SUCCESS)
    {
        ASSERT_THROW_HIPDNN_STATUS(resourceManager.getApplicableEngineIds(&mockGraphDesc),
                                   testCase.expectedThrow);
        return;
    }

    auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
    EXPECT_EQ(engineIds, testCase.expectedEngineIds);
}

INSTANTIATE_TEST_SUITE_P(
    SinglePlugin,
    TestEnginePluginResourceManagerApplicabilityFilter,
    ::testing::Values(
        ApplicabilityFilterCase{"BaselineExcludedForOverrideGraph",
                                hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE,
                                OverrideFlagReadMode::RETURNS_TRUE,
                                -1,
                                {}},
        ApplicabilityFilterCase{"BaselineIncludedForRegularGraph",
                                hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE,
                                OverrideFlagReadMode::RETURNS_FALSE,
                                -1,
                                {100}},
        ApplicabilityFilterCase{"BaselineIncludedWhenOverrideFlagUnsupported",
                                hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE,
                                OverrideFlagReadMode::THROWS_NOT_SUPPORTED,
                                -1,
                                {100}},
        ApplicabilityFilterCase{"BaselineIncludedWhenOverrideFlagCountZero",
                                hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE,
                                OverrideFlagReadMode::RETURNS_COUNT_ZERO,
                                -1,
                                {100}},
        ApplicabilityFilterCase{"OverrideFlagReadFailurePropagates",
                                hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE,
                                OverrideFlagReadMode::THROWS_INTERNAL_ERROR,
                                -1,
                                {},
                                HIPDNN_STATUS_INTERNAL_ERROR},
        ApplicabilityFilterCase{"OverrideCapableIncludedForOverrideGraph",
                                hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION,
                                OverrideFlagReadMode::RETURNS_TRUE,
                                1,
                                {100}},
        ApplicabilityFilterCase{"OverrideVersionWithoutSymbolExcludedForOverrideGraph",
                                hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION,
                                OverrideFlagReadMode::RETURNS_TRUE,
                                0,
                                {}},
        ApplicabilityFilterCase{"OverrideCapableIncludedForRegularGraph",
                                hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION,
                                OverrideFlagReadMode::RETURNS_FALSE,
                                -1,
                                {100}}),
    [](const auto& info) { return std::string(info.param.name); });

// Two plugins at K_OVERRIDE_EXECUTE_MIN_API_VERSION both contribute engines for
// an override-flag graph; the filter does not short-circuit.
TEST(TestEnginePluginResourceManager,
     ApplicabilityFilterIncludesAllOverrideCapablePluginsForOverrideGraph)
{
    auto pluginA = std::make_shared<MockEnginePlugin>();
    auto pluginB = std::make_shared<MockEnginePlugin>();
    std::vector<std::shared_ptr<EnginePlugin>> plugins{pluginA, pluginB};
    auto pluginManager = std::make_shared<MockEnginePluginManager>();

    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*pluginA, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xaaaaaaaa)));
    EXPECT_CALL(*pluginA, getAllEngineIds()).WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*pluginA, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION));
    EXPECT_CALL(*pluginA, destroyHandle(hipdnnEnginePluginHandle_t(0xaaaaaaaa)));

    EXPECT_CALL(*pluginB, createHandle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xbbbbbbbb)));
    EXPECT_CALL(*pluginB, getAllEngineIds()).WillOnce(::testing::Return(std::vector<int64_t>{200}));
    EXPECT_CALL(*pluginB, apiVersion())
        .WillRepeatedly(::testing::Return(hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION));
    EXPECT_CALL(*pluginB, destroyHandle(hipdnnEnginePluginHandle_t(0xbbbbbbbb)));

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};
    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));
    programOverrideFlag(mockGraphDesc, /*flag=*/true);

    EXPECT_CALL(*pluginA, getApplicableEngineIds(hipdnnEnginePluginHandle_t(0xaaaaaaaa), _))
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*pluginB, getApplicableEngineIds(hipdnnEnginePluginHandle_t(0xbbbbbbbb), _))
        .WillOnce(::testing::Return(std::vector<int64_t>{200}));
    EXPECT_CALL(*pluginA, hasOverrideExecute()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*pluginB, hasOverrideExecute()).WillOnce(::testing::Return(true));

    const EnginePluginResourceManager resourceManager(pluginManager);
    auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
    ASSERT_EQ(engineIds.size(), 2);
    // Order of plugins iteration is unordered_map-defined; assert as set.
    const std::set<int64_t> idSet(engineIds.begin(), engineIds.end());
    EXPECT_EQ(idSet, (std::set<int64_t>{100, 200}));
}

// With no plugins loaded, an override-flag graph yields an empty applicable
// engine list; the caller surfaces NOT_SUPPORTED.
TEST(TestEnginePluginResourceManager, ApplicabilityFilterEmptyWhenNoPluginsLoaded)
{
    std::vector<std::shared_ptr<EnginePlugin>> plugins;
    auto pluginManager = std::make_shared<MockEnginePluginManager>();
    EXPECT_CALL(*pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(plugins));

    MockGraphDescriptor mockGraphDesc;
    const hipdnnPluginConstData_t fakeSerializedData
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};
    EXPECT_CALL(mockGraphDesc, getSerializedGraph())
        .WillOnce(::testing::Return(fakeSerializedData));
    programOverrideFlag(mockGraphDesc, /*flag=*/true);

    const EnginePluginResourceManager resourceManager(pluginManager);
    auto engineIds = resourceManager.getApplicableEngineIds(&mockGraphDesc);
    EXPECT_TRUE(engineIds.empty());
}

namespace
{

// Common mock setup for executeOpGraph dispatch tests.
struct DispatchHarness
{
    std::shared_ptr<MockEnginePlugin> plugin;
    std::shared_ptr<MockEnginePluginManager> pluginManager;
    std::vector<std::shared_ptr<EnginePlugin>> pluginsList;
    std::unique_ptr<HipdnnBackendDescriptor> executionPlanWrapper;
    std::unique_ptr<HipdnnBackendDescriptor> variantWrapper;
    std::shared_ptr<MockExecutionPlanDescriptor> mockExecutionPlan;
    std::shared_ptr<MockVariantDescriptor> mockVariantPack;
};

DispatchHarness makeDispatchHarness(int64_t engineId,
                                    hipdnnEnginePluginHandle_t handle,
                                    hipdnnEnginePluginExecutionContext_t execCtx)
{
    DispatchHarness h;
    h.plugin = std::make_shared<MockEnginePlugin>();
    h.pluginsList = {h.plugin};
    h.pluginManager = std::make_shared<MockEnginePluginManager>();

    h.executionPlanWrapper = createDescriptor<MockExecutionPlanDescriptor>();
    h.variantWrapper = createDescriptor<MockVariantDescriptor>();

    h.mockExecutionPlan = MockDescriptorUtility::asDescriptorUnsafe<MockExecutionPlanDescriptor>(
        h.executionPlanWrapper.get());
    h.mockVariantPack
        = MockDescriptorUtility::asDescriptorUnsafe<MockVariantDescriptor>(h.variantWrapper.get());

    EXPECT_CALL(*h.pluginManager, getPlugins()).WillOnce(::testing::ReturnRef(h.pluginsList));
    EXPECT_CALL(*h.plugin, createHandle()).WillOnce(::testing::Return(handle));
    EXPECT_CALL(*h.plugin, getAllEngineIds())
        .WillOnce(::testing::Return(std::vector<int64_t>{engineId}));
    EXPECT_CALL(*h.plugin, destroyHandle(handle));

    EXPECT_CALL(*h.mockExecutionPlan, isFinalized()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*h.mockVariantPack, isFinalized()).WillOnce(::testing::Return(true));

    EXPECT_CALL(*h.mockExecutionPlan, getEngineId()).WillOnce(::testing::Return(engineId));
    // getExecutionContext() is called only on the successful dispatch path (not when dispatch
    // throws before reaching it, e.g. safety-net or bad-param tests).
    EXPECT_CALL(*h.mockExecutionPlan, getExecutionContext())
        .WillRepeatedly(::testing::Return(execCtx));
    return h;
}

enum class DispatchExpectedPath
{
    LEGACY_EXECUTE,
    THROW_BEFORE_EXECUTE
};

struct DispatchCase
{
    const char* name;
    int hasOverrideExecute; // -1: not queried, 0: queried false, 1: queried true.
    bool planOverrideShapeEnabled = true;
    std::vector<int64_t> overrideUniqueIds;
    std::vector<int64_t> overrideShapes;
    std::vector<int64_t> overrideStrides;
    std::vector<int64_t> overrideLengths;
    DispatchExpectedPath expectedPath;
    hipdnnStatus_t expectedThrow = HIPDNN_STATUS_SUCCESS;
    std::string_view apiVersion = hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION;
};

} // namespace

class TestEnginePluginResourceManagerDispatchMatrix : public ::testing::TestWithParam<DispatchCase>
{
};

TEST_P(TestEnginePluginResourceManagerDispatchMatrix, RoutesOrRejectsOverrideDispatch)
{
    const auto& testCase = GetParam();
    auto h = makeDispatchHarness(/*engineId=*/100,
                                 hipdnnEnginePluginHandle_t(0xdeadbeef),
                                 hipdnnEnginePluginExecutionContext_t(0xcafebabe));

    std::vector<int64_t> tensorIds{1, 2, 3};
    std::vector<const void*> dataPtrs{reinterpret_cast<void*>(0x1000),
                                      reinterpret_cast<void*>(0x2000),
                                      reinterpret_cast<void*>(0x3000)};
    std::vector<int64_t> overrideUniqueIds = testCase.overrideUniqueIds;
    std::vector<int64_t> overrideShapes = testCase.overrideShapes;
    std::vector<int64_t> overrideStrides = testCase.overrideStrides;
    std::vector<int64_t> overrideLengths = testCase.overrideLengths;

    EXPECT_CALL(*h.mockVariantPack, getWorkspace())
        .WillOnce(::testing::Return(reinterpret_cast<void*>(0x4000)));
    EXPECT_CALL(*h.mockVariantPack, getTensorIds()).WillOnce(::testing::ReturnRef(tensorIds));
    EXPECT_CALL(*h.mockVariantPack, getDataPointers()).WillOnce(::testing::ReturnRef(dataPtrs));
    EXPECT_CALL(*h.mockVariantPack, getOverrideUniqueIds())
        .WillOnce(::testing::ReturnRef(overrideUniqueIds));
    EXPECT_CALL(*h.mockVariantPack, getOverrideShapes())
        .WillOnce(::testing::ReturnRef(overrideShapes));
    EXPECT_CALL(*h.mockVariantPack, getOverrideStrides())
        .WillOnce(::testing::ReturnRef(overrideStrides));
    EXPECT_CALL(*h.mockVariantPack, getOverrideLengths())
        .WillOnce(::testing::ReturnRef(overrideLengths));

    const bool hasOverrides = !overrideUniqueIds.empty() || !overrideShapes.empty()
                              || !overrideStrides.empty() || !overrideLengths.empty();
    if(hasOverrides)
    {
        EXPECT_CALL(*h.mockExecutionPlan, isOverrideShapeEnabled())
            .WillOnce(::testing::Return(testCase.planOverrideShapeEnabled));
    }

    if(hasOverrides && testCase.planOverrideShapeEnabled && testCase.hasOverrideExecute >= 0)
    {
        EXPECT_CALL(*h.plugin, hasOverrideExecute())
            .WillOnce(::testing::Return(testCase.hasOverrideExecute == 1));
    }
    else
    {
        EXPECT_CALL(*h.plugin, hasOverrideExecute()).Times(0);
    }

    if(hasOverrides && testCase.planOverrideShapeEnabled && testCase.hasOverrideExecute == 1)
    {
        EXPECT_CALL(*h.plugin, apiVersion()).WillOnce(::testing::Return(testCase.apiVersion));
    }
    else
    {
        EXPECT_CALL(*h.plugin, apiVersion()).Times(0);
    }

    EXPECT_CALL(*h.plugin, executeOpGraphWithOverrides(_, _, _, _, _, _, _, _, _, _)).Times(0);

    if(testCase.expectedPath == DispatchExpectedPath::LEGACY_EXECUTE)
    {
        EXPECT_CALL(*h.plugin,
                    executeOpGraph(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                   hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                   reinterpret_cast<void*>(0x4000),
                                   ::testing::NotNull(),
                                   static_cast<uint32_t>(3)));
    }
    else
    {
        EXPECT_CALL(*h.plugin, executeOpGraph(_, _, _, _, _)).Times(0);
    }

    const EnginePluginResourceManager resourceManager(h.pluginManager);
    if(testCase.expectedThrow != HIPDNN_STATUS_SUCCESS)
    {
        ASSERT_THROW_HIPDNN_STATUS(
            resourceManager.executeOpGraph(h.executionPlanWrapper.get(), h.variantWrapper.get()),
            testCase.expectedThrow);
        return;
    }

    resourceManager.executeOpGraph(h.executionPlanWrapper.get(), h.variantWrapper.get());
}

INSTANTIATE_TEST_SUITE_P(
    OverrideDispatch,
    TestEnginePluginResourceManagerDispatchMatrix,
    ::testing::Values(DispatchCase{"VersionLiarPluginRejected",
                                   /*hasOverrideExecute=*/0,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {2, 3, 4},
                                   {12, 4, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_NOT_SUPPORTED},
                      DispatchCase{"NoOverridesRoutesToLegacyEntry",
                                   /*hasOverrideExecute=*/-1,
                                   /*planOverrideShapeEnabled=*/false,
                                   {},
                                   {},
                                   {},
                                   {},
                                   DispatchExpectedPath::LEGACY_EXECUTE},
                      DispatchCase{"MismatchedOverrideLengthsRejected",
                                   /*hasOverrideExecute=*/-1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1, 2},
                                   {2, 3, 4, 5},
                                   {12, 4, 1, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_BAD_PARAM},
                      DispatchCase{"OverrideShapesFlatCountMismatchRejected",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {2, 3},
                                   {12, 4, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_BAD_PARAM},
                      DispatchCase{"OverrideStridesFlatCountMismatchRejected",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {2, 3, 4},
                                   {12, 4, 1, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_BAD_PARAM},
                      DispatchCase{"ZeroOverrideLengthRejected",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {},
                                   {},
                                   {0},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND},
                      DispatchCase{"OverrideLengthExceedsUint32Rejected",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {},
                                   {},
                                   {static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND},
                      DispatchCase{"OverrideMetadataRejectedWhenPlanNotOverrideEnabled",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/false,
                                   {1},
                                   {2, 3, 4},
                                   {12, 4, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_NOT_SUPPORTED},
                      DispatchCase{"OldApiVersionWithOverrideSymbolRejected",
                                   /*hasOverrideExecute=*/1,
                                   /*planOverrideShapeEnabled=*/true,
                                   {1},
                                   {2, 3, 4},
                                   {12, 4, 1},
                                   {3},
                                   DispatchExpectedPath::THROW_BEFORE_EXECUTE,
                                   HIPDNN_STATUS_NOT_SUPPORTED,
                                   hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE}),
    [](const auto& info) { return std::string(info.param.name); });

// When the variant pack carries override tensors and the plugin exports the
// override symbol, dispatch must:
//   - call `executeOpGraphWithOverrides` (NOT `executeOpGraph`),
//   - reconstruct per-UID shape/stride pointer arrays from the flat buffers
//     (3 tensors of ranks 3, 4, 4 over an 11-element buffer),
//   - narrow lengths from int64 to uint32,
//   - read each variant-pack accessor exactly once (lifetime / mutation safety).
TEST(TestEnginePluginResourceManager, DispatchRoutesToOverrideEntryWithReconstructedPtrArrays)
{
    auto h = makeDispatchHarness(/*engineId=*/100,
                                 hipdnnEnginePluginHandle_t(0xdeadbeef),
                                 hipdnnEnginePluginExecutionContext_t(0xcafebabe));

    std::vector<int64_t> tensorIds{1, 2, 3};
    std::vector<const void*> dataPtrs{reinterpret_cast<void*>(0x1000),
                                      reinterpret_cast<void*>(0x2000),
                                      reinterpret_cast<void*>(0x3000)};

    // Three tensors of ranks {3, 4, 4}: flat-buffer offsets are {0, 3, 7}.
    std::vector<int64_t> overrideUniqueIds{1, 2, 3};
    std::vector<int64_t> overrideLengths{3, 4, 4};
    std::vector<int64_t> overrideShapes{// tensor 1 (rank 3)
                                        2,
                                        3,
                                        4,
                                        // tensor 2 (rank 4)
                                        1,
                                        2,
                                        3,
                                        4,
                                        // tensor 3 (rank 4)
                                        5,
                                        6,
                                        7,
                                        8};
    std::vector<int64_t> overrideStrides{// tensor 1
                                         12,
                                         4,
                                         1,
                                         // tensor 2
                                         24,
                                         12,
                                         4,
                                         1,
                                         // tensor 3
                                         336,
                                         56,
                                         8,
                                         1};

    EXPECT_CALL(*h.mockVariantPack, getWorkspace())
        .WillOnce(::testing::Return(reinterpret_cast<void*>(0x4000)));
    EXPECT_CALL(*h.mockVariantPack, getTensorIds()).WillOnce(::testing::ReturnRef(tensorIds));
    EXPECT_CALL(*h.mockVariantPack, getDataPointers()).WillOnce(::testing::ReturnRef(dataPtrs));

    // Each accessor is read exactly once by the dispatch wrapper. WillOnce
    // here doubles as a strict-arity check.
    EXPECT_CALL(*h.mockVariantPack, getOverrideUniqueIds())
        .WillOnce(::testing::ReturnRef(overrideUniqueIds));
    EXPECT_CALL(*h.mockVariantPack, getOverrideShapes())
        .WillOnce(::testing::ReturnRef(overrideShapes));
    EXPECT_CALL(*h.mockVariantPack, getOverrideStrides())
        .WillOnce(::testing::ReturnRef(overrideStrides));
    EXPECT_CALL(*h.mockVariantPack, getOverrideLengths())
        .WillOnce(::testing::ReturnRef(overrideLengths));

    EXPECT_CALL(*h.plugin, hasOverrideExecute()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*h.plugin, apiVersion())
        .WillOnce(::testing::Return(hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION));
    EXPECT_CALL(*h.mockExecutionPlan, isOverrideShapeEnabled()).WillOnce(::testing::Return(true));

    // Capture the call to verify length narrowing and pointer reconstruction.
    EXPECT_CALL(*h.plugin,
                executeOpGraphWithOverrides(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                            hipdnnEnginePluginExecutionContext_t(0xcafebabe),
                                            reinterpret_cast<void*>(0x4000),
                                            ::testing::NotNull(), // deviceBuffers
                                            static_cast<uint32_t>(3), // numDeviceBuffers
                                            static_cast<uint32_t>(3), // numOverrides
                                            ::testing::_, // overrideUniqueIds
                                            ::testing::_, // overrideLengths (uint32)
                                            ::testing::_, // overrideShapes (ptr-array)
                                            ::testing::_)) // overrideStrides (ptr-array)
        .WillOnce(::testing::Invoke([&overrideUniqueIds, &overrideShapes, &overrideStrides](
                                        hipdnnEnginePluginHandle_t,
                                        hipdnnEnginePluginExecutionContext_t,
                                        void*,
                                        const hipdnnPluginDeviceBuffer_t*,
                                        uint32_t,
                                        uint32_t numOverrides,
                                        const int64_t* uniqueIds,
                                        const uint32_t* lengths,
                                        const int64_t* const* shapesPerUid,
                                        const int64_t* const* stridesPerUid) {
            ASSERT_EQ(numOverrides, 3u);
            // Lengths are narrowed to uint32 with values matching the int64 source.
            EXPECT_EQ(lengths[0], 3u);
            EXPECT_EQ(lengths[1], 4u);
            EXPECT_EQ(lengths[2], 4u);
            // unique ids forwarded by-pointer; values match.
            EXPECT_EQ(uniqueIds[0], overrideUniqueIds[0]);
            EXPECT_EQ(uniqueIds[1], overrideUniqueIds[1]);
            EXPECT_EQ(uniqueIds[2], overrideUniqueIds[2]);
            // Per-UID shape pointers reference offsets {0, 3, 7} into the flat
            // shape buffer owned by the variant pack.
            EXPECT_EQ(shapesPerUid[0], overrideShapes.data() + 0);
            EXPECT_EQ(shapesPerUid[1], overrideShapes.data() + 3);
            EXPECT_EQ(shapesPerUid[2], overrideShapes.data() + 7);
            // Stride pointers use the same offsets.
            EXPECT_EQ(stridesPerUid[0], overrideStrides.data() + 0);
            EXPECT_EQ(stridesPerUid[1], overrideStrides.data() + 3);
            EXPECT_EQ(stridesPerUid[2], overrideStrides.data() + 7);
            // Spot-check a value through each pointer to confirm the slice
            // really sees the per-tensor data (lifetime safety: the data
            // is still live at dispatch time).
            EXPECT_EQ(shapesPerUid[1][0], 1);
            EXPECT_EQ(shapesPerUid[2][3], 8);
            EXPECT_EQ(stridesPerUid[1][3], 1);
        }));

    // Legacy entry must NOT be reached.
    EXPECT_CALL(*h.plugin, executeOpGraph(_, _, _, _, _)).Times(0);

    const EnginePluginResourceManager resourceManager(h.pluginManager);
    resourceManager.executeOpGraph(h.executionPlanWrapper.get(), h.variantWrapper.get());
}

// =============================================================================
// `PluginBase` metadata.
// =============================================================================
TEST(TestPluginBase, MockPluginCachedNameUsesDeterministicFallback)
{
    const MockEnginePlugin plugin;

    EXPECT_EQ(plugin.cachedName(), "mock_plugin");
}

TEST(TestPluginBase, ParsedApiVersionReturnsNulloptForMalformedString)
{
    const MockEnginePlugin plugin;

    EXPECT_CALL(plugin, apiVersion())
        .Times(1)
        .WillOnce(::testing::Return(std::string_view{"not.a.version"}));

    const auto parsed = plugin.parsedApiVersion();
    EXPECT_FALSE(parsed.has_value()) << "Malformed version string must yield nullopt.";
}

TEST(TestPluginBase, ParsedApiVersionReturnsParsedValueForWellFormedString)
{
    const MockEnginePlugin plugin;

    EXPECT_CALL(plugin, apiVersion())
        .Times(1)
        .WillOnce(::testing::Return(hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION));

    const auto parsed = plugin.parsedApiVersion();
    ASSERT_TRUE(parsed.has_value()) << "Well-formed version string must parse successfully.";
    EXPECT_EQ(parsed->major, 1);
    EXPECT_EQ(parsed->minor, 1);
    EXPECT_EQ(parsed->patch, 0);
}

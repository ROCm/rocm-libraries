// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "mocks/MockEngine.hpp"
#include "mocks/MockHipdnnHipKernelContext.hpp"

using namespace hip_kernel_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using ::testing::Return;

TEST(TestHipKernelEngineManager, ReturnsApplicableEngineIds)
{
    std::set<std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                        HipdnnHipKernelSettings,
                                                        HipdnnHipKernelContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnHipKernelHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_EQ(applicable[0], 1);
}

TEST(TestHipKernelEngineManager, ReturnsMultipleApplicableEngineIds)
{
    std::set<std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                        HipdnnHipKernelSettings,
                                                        HipdnnHipKernelContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnHipKernelHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 1) != applicable.end());
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 2) != applicable.end());
}

TEST(TestHipKernelEngineManager, ReturnsNoApplicableEngineIds)
{
    std::set<std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                        HipdnnHipKernelSettings,
                                                        HipdnnHipKernelContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnHipKernelHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_TRUE(applicable.empty());
}

TEST(TestHipKernelEngineManager, ReturnsEngineDetails)
{
    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;

    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = reinterpret_cast<const void*>(0x12345678);
    engineDetails.size = 200;
    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine, getDetails(::testing::_, ::testing::_, ::testing::_))
        .WillOnce([&engineDetails](HipdnnHipKernelHandle& handle,
                                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph,
                                   hipdnnPluginConstData_t& out) {
            (void)handle;
            (void)graph;
            out.ptr = engineDetails.ptr;
            out.size = engineDetails.size;
        });

    manager.addEngine(std::move(mockEngine));

    MockGraph mockGraph;
    HipdnnHipKernelHandle dummyHandle = {};
    hipdnnPluginConstData_t details;
    manager.getEngineDetails(dummyHandle, mockGraph, 1, details);

    EXPECT_EQ(details.ptr, engineDetails.ptr);
    EXPECT_EQ(details.size, engineDetails.size);
}

TEST(TestHipKernelEngineManager, ThrowsOnInvalidEngineId)
{
    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;

    MockGraph mockGraph;
    hipdnnPluginConstData_t engineDetails;

    HipdnnHipKernelHandle dummyHandle = {};
    EXPECT_THROW(manager.getEngineDetails(dummyHandle, mockGraph, 999, engineDetails),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelEngineManager, GetWorkspaceSizeReturnsCorrectValue)
{
    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;

    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(42));
    HipdnnHipKernelHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    EXPECT_CALL(mockEngineConfig, engineId()).WillRepeatedly(Return(42));
    EXPECT_CALL(*mockEngine, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Return(4096));

    manager.addEngine(std::move(mockEngine));

    size_t workspaceSize = manager.getMaxWorkspaceSize(dummyHandle, mockGraph, mockEngineConfig);
    EXPECT_EQ(workspaceSize, 4096);
}

TEST(TestHipKernelEngineManager, GetWorkspaceSizeThrowsOnInvalidEngineId)
{
    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    HipdnnHipKernelHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    EXPECT_CALL(mockEngineConfig, engineId()).WillRepeatedly(Return(999));

    EXPECT_THROW(manager.getMaxWorkspaceSize(dummyHandle, mockGraph, mockEngineConfig),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelEngineManager, InitializeExecutionContextCallsEngine)
{
    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(7));
    EXPECT_CALL(*mockEngine,
                initializeExecutionContext(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    manager.addEngine(std::move(mockEngine));
    HipdnnHipKernelHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    ON_CALL(mockEngineConfig, engineId()).WillByDefault(Return(7));
    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    MockHipdnnHipKernelContext execCtx;

    manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx);
}

TEST(TestHipKernelEngineManager, InitializeExecutionContextThrowsOnInvalidEngineId)
{
    MockHipdnnHipKernelContext execCtx;
    hipdnn_plugin_sdk::
        EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
            manager;
    HipdnnHipKernelHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;

    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    EXPECT_THROW(
        manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

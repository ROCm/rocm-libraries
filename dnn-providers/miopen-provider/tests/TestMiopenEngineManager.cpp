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
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "mocks/MockEngine.hpp"
#include "mocks/MockHipdnnMiopenContext.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using ::testing::Return;

TEST(TestMiopenEngineManager, ReturnsApplicableEngineIds)
{
    SKIP_IF_NO_DEVICES();

    std::set<std::unique_ptr<
        hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnMiopenHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_EQ(applicable[0], 1);
}

TEST(TestMiopenEngineManager, ReturnsMultipleApplicableEngineIds)
{
    SKIP_IF_NO_DEVICES();

    std::set<std::unique_ptr<
        hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnMiopenHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 1) != applicable.end());
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 2) != applicable.end());
}

TEST(TestMiopenEngineManager, ReturnsNoApplicableEngineIds)
{
    SKIP_IF_NO_DEVICES();

    std::set<std::unique_ptr<
        hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>>
        engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnMiopenHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_TRUE(applicable.empty());
}

TEST(TestMiopenEngineManager, ReturnsEngineDetails)
{
    SKIP_IF_NO_DEVICES();

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;

    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = reinterpret_cast<const void*>(0x12345678);
    engineDetails.size = 200;
    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine, getDetails(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(
            [&engineDetails](HipdnnMiopenHandle& handle,
                             const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                             hipdnnPluginConstData_t& out) {
                (void)handle;
                (void)graph;
                out.ptr = engineDetails.ptr;
                out.size = engineDetails.size;
            });

    manager.addEngine(std::move(mockEngine));

    MockGraph mockGraph;
    HipdnnMiopenHandle dummyHandle = {};
    hipdnnPluginConstData_t details;
    manager.getEngineDetails(dummyHandle, mockGraph, 1, details);

    EXPECT_EQ(details.ptr, engineDetails.ptr);
    EXPECT_EQ(details.size, engineDetails.size);
}

TEST(TestMiopenEngineManager, ThrowsOnInvalidEngineId)
{
    SKIP_IF_NO_DEVICES();

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;

    MockGraph mockGraph;
    hipdnnPluginConstData_t engineDetails;

    HipdnnMiopenHandle dummyHandle = {};
    EXPECT_THROW(manager.getEngineDetails(dummyHandle, mockGraph, 999, engineDetails),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenEngineManager, GetWorkspaceSizeReturnsCorrectValue)
{
    SKIP_IF_NO_DEVICES();

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;

    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(42));
    HipdnnMiopenHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    EXPECT_CALL(mockEngineConfig, engineId()).WillRepeatedly(Return(42));
    EXPECT_CALL(*mockEngine, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Return(4096));

    manager.addEngine(std::move(mockEngine));

    size_t workspaceSize = manager.getMaxWorkspaceSize(dummyHandle, mockGraph, mockEngineConfig);
    EXPECT_EQ(workspaceSize, 4096);
}

TEST(TestMiopenEngineManager, GetWorkspaceSizeThrowsOnInvalidEngineId)
{
    SKIP_IF_NO_DEVICES();

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    HipdnnMiopenHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    EXPECT_CALL(mockEngineConfig, engineId()).WillRepeatedly(Return(999));

    EXPECT_THROW(manager.getMaxWorkspaceSize(dummyHandle, mockGraph, mockEngineConfig),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenEngineManager, InitializeExecutionContextCallsEngine)
{
    SKIP_IF_NO_DEVICES();

    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(7));
    EXPECT_CALL(*mockEngine,
                initializeExecutionContext(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    manager.addEngine(std::move(mockEngine));
    HipdnnMiopenHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    ON_CALL(mockEngineConfig, engineId()).WillByDefault(Return(7));
    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    MockHipdnnMiopenContext execCtx;

    manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx);
}

TEST(TestMiopenEngineManager, InitializeExecutionContextThrowsOnInvalidEngineId)
{
    SKIP_IF_NO_DEVICES();

    MockHipdnnMiopenContext execCtx;
    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
        manager;
    HipdnnMiopenHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;

    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    EXPECT_THROW(
        manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

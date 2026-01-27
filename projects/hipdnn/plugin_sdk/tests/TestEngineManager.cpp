// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngine.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/MockPlan.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

// Define test handle and execution context structs for testing
struct HipdnnEnginePluginHandle
{
};

struct HipdnnEnginePluginExecutionContext
{
};

namespace
{

std::unique_ptr<NiceMock<MockEngine>>
    createMockEngine(int64_t engineId, bool applicable, size_t workspaceSize = 1024)
{
    auto engine = std::make_unique<NiceMock<MockEngine>>();
    ON_CALL(*engine, id()).WillByDefault(Return(engineId));
    ON_CALL(*engine, isApplicable(_, _)).WillByDefault(Return(applicable));
    ON_CALL(*engine, getMaxWorkspaceSize(_, _)).WillByDefault(Return(workspaceSize));
    return engine;
}

} // namespace

TEST(TestEngineManager, InitiallyHasNoEngines)
{
    EngineManager manager;
    auto engineIds = manager.getAllEngineIds();
    EXPECT_TRUE(engineIds.empty());
}

TEST(TestEngineManager, AddEngineRegistersEngine)
{
    EngineManager manager;
    manager.addEngine(createMockEngine(1, true));

    auto engineIds = manager.getAllEngineIds();
    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], 1);
}

TEST(TestEngineManager, AddMultipleEngines)
{
    EngineManager manager;
    manager.addEngine(createMockEngine(1, true));
    manager.addEngine(createMockEngine(2, false));
    manager.addEngine(createMockEngine(3, true));

    auto engineIds = manager.getAllEngineIds();
    EXPECT_EQ(engineIds.size(), 3u);
}

TEST(TestEngineManager, GetApplicableEngineIdsFiltersCorrectly)
{
    EngineManager manager;
    manager.addEngine(createMockEngine(1, true));
    manager.addEngine(createMockEngine(2, false));
    manager.addEngine(createMockEngine(3, true));

    HipdnnEnginePluginHandle handle;
    NiceMock<MockGraph> mockGraph;
    auto applicableIds = manager.getApplicableEngineIds(handle, mockGraph);

    EXPECT_EQ(applicableIds.size(), 2u);
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 1) != applicableIds.end());
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 3) != applicableIds.end());
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 2) == applicableIds.end());
}

TEST(TestEngineManager, GetWorkspaceSizeReturnsEngineWorkspace)
{
    EngineManager manager;
    manager.addEngine(createMockEngine(1, true, 2048));

    HipdnnEnginePluginHandle handle;
    NiceMock<MockGraph> mockGraph;
    auto workspaceSize = manager.getWorkspaceSize(handle, 1, mockGraph);
    EXPECT_EQ(workspaceSize, 2048u);
}

TEST(TestEngineManager, GetWorkspaceSizeThrowsForUnknownEngine)
{
    EngineManager manager;
    manager.addEngine(createMockEngine(1, true));

    HipdnnEnginePluginHandle handle;
    NiceMock<MockGraph> mockGraph;
    EXPECT_THROW(manager.getWorkspaceSize(handle, 999, mockGraph), HipdnnPluginException);
}

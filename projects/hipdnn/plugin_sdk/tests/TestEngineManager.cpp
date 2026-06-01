// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;
using ::testing::NiceMock;
using ::testing::Return;

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

namespace
{

// Test engine that inherits from IEngine with configurable behavior
class TestEngine : public IEngine<TestHandle, TestSettings, TestContext>
{
public:
    TestEngine(int64_t engineId, bool applicable, size_t workspaceSize = 1024)
        : _id(engineId)
        , _applicable(applicable)
        , _workspaceSize(workspaceSize)
    {
    }

    int64_t id() const override
    {
        return _id;
    }

    bool isApplicable(
        TestHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return _applicable;
    }

    void getDetails(TestHandle& /*handle*/,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                    hipdnnPluginConstData_t& /*detailsOut*/) const override
    {
    }

    size_t getMaxWorkspaceSize(
        const TestHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/)
        const override
    {
        return _workspaceSize;
    }

    void initializeExecutionContext(
        const TestHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
        TestContext& /*executionContext*/) const override
    {
    }

private:
    int64_t _id;
    bool _applicable;
    size_t _workspaceSize;
};

// Define type alias for readability
using TestEngineManager = EngineManager<TestHandle, TestSettings, TestContext>;

std::unique_ptr<TestEngine>
    createTestEngine(int64_t engineId, bool applicable, size_t workspaceSize = 1024)
{
    return std::make_unique<TestEngine>(engineId, applicable, workspaceSize);
}

} // namespace

TEST(TestEngineManager, InitiallyHasNoEngines)
{
    const TestEngineManager manager;
    auto engineIds = manager.getAllEngineIds();
    EXPECT_TRUE(engineIds.empty());
}

TEST(TestEngineManager, AddEngineRegistersEngine)
{
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true));

    auto engineIds = manager.getAllEngineIds();
    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], 1);
}

TEST(TestEngineManager, AddMultipleEngines)
{
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true));
    manager.addEngine(createTestEngine(2, false));
    manager.addEngine(createTestEngine(3, true));

    auto engineIds = manager.getAllEngineIds();
    EXPECT_EQ(engineIds.size(), 3u);
}

TEST(TestEngineManager, GetApplicableEngineIdsFiltersCorrectly)
{
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true));
    manager.addEngine(createTestEngine(2, false));
    manager.addEngine(createTestEngine(3, true));

    TestHandle handle;
    const NiceMock<MockGraph> mockGraph;
    auto applicableIds = manager.getApplicableEngineIds(handle, mockGraph);

    EXPECT_EQ(applicableIds.size(), 2u);
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 1) != applicableIds.end());
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 3) != applicableIds.end());
    EXPECT_TRUE(std::find(applicableIds.begin(), applicableIds.end(), 2) == applicableIds.end());
}

TEST(TestEngineManager, GetWorkspaceSizeReturnsEngineWorkspace)
{
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true, 2048));

    const TestHandle handle;
    const NiceMock<MockGraph> mockGraph;
    const NiceMock<MockEngineConfig> mockConfig;
    ON_CALL(mockConfig, engineId()).WillByDefault(Return(1));

    auto workspaceSize = manager.getMaxWorkspaceSize(handle, mockGraph, mockConfig);
    EXPECT_EQ(workspaceSize, 2048u);
}

TEST(TestEngineManager, GetWorkspaceSizeThrowsForUnknownEngine)
{
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true));

    const TestHandle handle;
    const NiceMock<MockGraph> mockGraph;
    const NiceMock<MockEngineConfig> mockConfig;
    ON_CALL(mockConfig, engineId()).WillByDefault(Return(999));

    EXPECT_THROW(manager.getMaxWorkspaceSize(handle, mockGraph, mockConfig), HipdnnPluginException);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;
using ::testing::NiceMock;
using ::testing::Return;

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

/// Two engines hashing onto one id is an authoring collision (RFC 0003); the manager keeps
/// the incumbent and logs the loser rather than dropping it silently. This is not the
/// descriptor rule -- RFC 0020 §10.2.1 drops every UED in a name collision before
/// construction -- but the backstop for collisions that rule can't see, e.g. two
/// hand-written engines.
TEST(TestEngineManager, AddEngineKeepsTheIncumbentOnDuplicateId)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true, 2048));
    manager.addEngine(createTestEngine(1, true, 4096));

    auto engineIds = manager.getAllEngineIds();
    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], 1);

    // The point of this assertion: the old bare `emplace` would also pass everything above.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "already registered"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "duplicate is discarded"));

    // The incumbent is the one still answering, not the arrival that lost the insert.
    const TestHandle handle;
    const NiceMock<MockGraph> mockGraph;
    const NiceMock<MockEngineConfig> mockConfig;
    ON_CALL(mockConfig, engineId()).WillByDefault(Return(1));
    EXPECT_EQ(manager.getMaxWorkspaceSize(handle, mockGraph, mockConfig), 2048u);
}

TEST(TestEngineManager, AddEngineNamesTheDiscardedDuplicateWhenTheCallerSuppliesAName)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    TestEngineManager manager;
    manager.addEngine(createTestEngine(1, true, 2048), "hipkernel:ConvFwd");
    manager.addEngine(createTestEngine(1, true, 4096), "hipkernel:ConvFwd");

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "hipkernel:ConvFwd"));

    // The name joins the id rather than replacing it: the id is what greps against the
    // admission logs.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "0x0000000000000001"));
}

TEST(TestEngineManager, AddEngineNamesTheDiscardedDuplicateFromTheRegistryWhenNoNameIsGiven)
{
    using hipdnn_data_sdk::utilities::formatEngineIdHex;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    TestEngineManager manager;
    manager.addEngine(createTestEngine(MIOPEN_ENGINE_ID, true, 2048));
    manager.addEngine(createTestEngine(MIOPEN_ENGINE_ID, true, 4096));

    // No name was declared, so this one comes from the static registry.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "MIOPEN_ENGINE"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, formatEngineIdHex(MIOPEN_ENGINE_ID)));
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

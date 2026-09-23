// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestSelectionHeuristic.cpp
 * @brief Unit tests for SelectionHeuristic RAII wrapper
 *
 * Tests the C++ facade that wraps hipdnnHeuristicPolicyDescriptor_t lifecycle
 * and provides clean API over the heuristic plugin C ABI.
 */

#include "heuristics/SelectionHeuristic.hpp"

#include "descriptors/mocks/MockHeuristicPlugin.hpp"
#include "descriptors/mocks/MockHeuristicPluginResourceManager.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

using namespace hipdnn_backend::heuristics;
using namespace hipdnn_backend::plugin;
using ::testing::NiceMock;

class TestSelectionHeuristic : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a mock plugin handle (just a non-null pointer for testing)
        _mockHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(this);
        _mockPlugin = std::make_unique<MockHeuristicPlugin>();
        _mockResourceManager = std::make_shared<NiceMock<MockHeuristicPluginResourceManager>>();
    }

    // Wires the mock resource manager to return _mockHandle / _mockPlugin.get()
    // for the test policy ID. Use AnyNumber so cleanup-time lookups in the
    // SelectionHeuristic destructor (and move ops) are also satisfied.
    void wireResourceManager()
    {
        EXPECT_CALL(*_mockResourceManager, getHeuristicHandleForPolicyId(_policyId))
            .WillRepeatedly(::testing::Return(_mockHandle));
        EXPECT_CALL(*_mockResourceManager, getPluginForPolicyId(_policyId))
            .WillRepeatedly(::testing::Return(_mockPlugin.get()));
    }

    hipdnnHeuristicHandle_t _mockHandle = nullptr;
    std::unique_ptr<MockHeuristicPlugin> _mockPlugin;
    std::shared_ptr<NiceMock<MockHeuristicPluginResourceManager>> _mockResourceManager;
    int64_t _policyId = 12345;
};

// ========== Constructor Tests ==========

TEST_F(TestSelectionHeuristic, ConstructorWithValidInputs)
{
    wireResourceManager();

    // Expect createPolicyDescriptor to be called
    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1234);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    // Should not throw
    const SelectionHeuristic heuristic(_mockResourceManager, _policyId);
}

TEST_F(TestSelectionHeuristic, ConstructorThrowsOnNullResourceManager)
{
    EXPECT_THROW(
        {
            SelectionHeuristic heuristic(nullptr, _policyId); // NOLINT(misc-const-correctness)
        },
        hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, ConstructorThrowsWhenPolicyHasNoHandle)
{
    // Manager reports no handle for this policy ID
    EXPECT_CALL(*_mockResourceManager, getHeuristicHandleForPolicyId(_policyId))
        .WillRepeatedly(::testing::Return(nullptr));

    EXPECT_THROW(
        { const SelectionHeuristic heuristic(_mockResourceManager, _policyId); },
        hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, ConstructorThrowsWhenPolicyHasNoPlugin)
{
    // Manager reports a handle but no plugin (defensive check)
    EXPECT_CALL(*_mockResourceManager, getHeuristicHandleForPolicyId(_policyId))
        .WillRepeatedly(::testing::Return(_mockHandle));
    EXPECT_CALL(*_mockResourceManager, getPluginForPolicyId(_policyId))
        .WillRepeatedly(::testing::Return(nullptr));

    EXPECT_THROW(
        { const SelectionHeuristic heuristic(_mockResourceManager, _policyId); },
        hipdnn_backend::HipdnnException);
}

// ========== Move Semantics Tests ==========

TEST_F(TestSelectionHeuristic, MoveConstructor)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    // Destroy should be called exactly once (when moved-to object is destroyed)
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    // Regression: the move must carry _inputEngineIds too, otherwise the
    // validator in getSortedEngineIds sees an empty candidate set and rejects
    // every legitimate plugin output with HIPDNN_STATUS_PLUGIN_ERROR.
    const std::vector<int64_t> inputIds = {1, 2, 3};
    const std::vector<int64_t> sortedIds = {3, 1, 2};
    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(sortedIds));

    {
        SelectionHeuristic heuristic1(_mockResourceManager, _policyId);
        heuristic1.setEngineIds(inputIds);

        // Move construct
        SelectionHeuristic heuristic2(std::move(heuristic1)); // NOLINT(misc-const-correctness)

        // heuristic2 should now own the descriptor and the input-ID candidate set
        EXPECT_EQ(heuristic2.getSortedEngineIds(), sortedIds);
        // heuristic1 should be empty (moved-from state)
    } // Both destructors called, but only heuristic2 has valid descriptor
}

TEST_F(TestSelectionHeuristic, MoveAssignment)
{
    wireResourceManager();

    auto mockDescriptor1 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1111);
    auto mockDescriptor2 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x2222);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor1))
        .WillOnce(::testing::Return(mockDescriptor2));

    // First descriptor destroyed during move assignment
    // Second descriptor destroyed when moved-to object is destroyed
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor1)).Times(1);
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor2)).Times(1);

    // Regression: the move must carry _inputEngineIds too, so the validator
    // in getSortedEngineIds still has the candidate set after the assignment.
    const std::vector<int64_t> inputIds = {10, 20, 30};
    const std::vector<int64_t> sortedIds = {30, 20, 10};
    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor2, ::testing::_, inputIds.size()))
        .Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor2))
        .WillOnce(::testing::Return(sortedIds));

    {
        SelectionHeuristic heuristic1(_mockResourceManager, _policyId);
        SelectionHeuristic heuristic2(_mockResourceManager, _policyId);
        heuristic2.setEngineIds(inputIds);

        // Move assign
        heuristic1 = std::move(heuristic2);

        // heuristic1 should now own mockDescriptor2 and heuristic2's input IDs
        // heuristic2 should be empty
        // mockDescriptor1 should have been destroyed
        EXPECT_EQ(heuristic1.getSortedEngineIds(), sortedIds);
    }
}

TEST_F(TestSelectionHeuristic, MoveAssignmentSelfAssignment)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x9999);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    {
        SelectionHeuristic heuristic(_mockResourceManager, _policyId);

        // Self-assignment should be safe (use reference to avoid warning)
        SelectionHeuristic& heuristicRef = heuristic;
        heuristic = std::move(heuristicRef);

        // Descriptor should still be valid
    }
}

// ========== API Tests ==========

TEST_F(TestSelectionHeuristic, SetEngineIds)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xAAAA);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    std::vector<int64_t> testEngineIds = {1, 2, 3, 4, 5};

    EXPECT_CALL(
        *_mockPlugin,
        setEngineIds(mockDescriptor, ::testing::Pointee(testEngineIds[0]), testEngineIds.size()))
        .Times(1);

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(testEngineIds);
}

TEST_F(TestSelectionHeuristic, SetSerializedGraph)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xBBBB);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    std::vector<uint8_t> graphData = {0x01, 0x02, 0x03};
    const hipdnnPluginConstData_t serializedGraph{graphData.data(), graphData.size()};

    EXPECT_CALL(*_mockPlugin, setSerializedGraph(mockDescriptor, &serializedGraph)).Times(1);

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setSerializedGraph(&serializedGraph);
}

TEST_F(TestSelectionHeuristic, FinalizeReturnsTrue)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xCCCC);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*_mockPlugin, finalize(mockDescriptor)).WillOnce(::testing::Return(true));

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    EXPECT_TRUE(heuristic.finalize());
}

TEST_F(TestSelectionHeuristic, FinalizeReturnsFalse)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xDDDD);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    EXPECT_CALL(*_mockPlugin, finalize(mockDescriptor)).WillOnce(::testing::Return(false));

    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    EXPECT_FALSE(heuristic.finalize());
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIds)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    // The plugin output must be a permutation/subset of the IDs we hand in
    // via setEngineIds — that's what SelectionHeuristic validates.
    const std::vector<int64_t> inputIds = {1, 2, 3, 4, 5};
    const std::vector<int64_t> expectedIds = {5, 4, 3, 2, 1};

    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(expectedIds));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(inputIds);
    auto result = heuristic.getSortedEngineIds();

    EXPECT_EQ(result, expectedIds);
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIdsRejectsFabricatedId)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    const std::vector<int64_t> inputIds = {1, 2, 3};
    // Plugin returns an ID (99) that wasn't in the candidate set.
    const std::vector<int64_t> badIds = {2, 99, 1};

    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(badIds));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(inputIds);
    EXPECT_THROW(heuristic.getSortedEngineIds(), hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIdsRejectsDuplicates)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    const std::vector<int64_t> inputIds = {1, 2, 3};
    const std::vector<int64_t> badIds = {1, 2, 2};

    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(badIds));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(inputIds);
    EXPECT_THROW(heuristic.getSortedEngineIds(), hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIdsRejectsOversizedOutput)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    const std::vector<int64_t> inputIds = {1, 2};
    const std::vector<int64_t> badIds = {1, 2, 3};

    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(badIds));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(inputIds);
    EXPECT_THROW(heuristic.getSortedEngineIds(), hipdnn_backend::HipdnnException);
}

TEST_F(TestSelectionHeuristic, GetSortedEngineIdsAcceptsProperSubset)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xEEEE);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    // Plugin may decline some candidates and return a strict subset.
    const std::vector<int64_t> inputIds = {1, 2, 3, 4, 5};
    const std::vector<int64_t> expectedIds = {3, 1};

    EXPECT_CALL(*_mockPlugin, setEngineIds(mockDescriptor, ::testing::_, inputIds.size())).Times(1);
    EXPECT_CALL(*_mockPlugin, getSortedEngineIds(mockDescriptor))
        .WillOnce(::testing::Return(expectedIds));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    heuristic.setEngineIds(inputIds);
    EXPECT_EQ(heuristic.getSortedEngineIds(), expectedIds);
}

// ========== Lifetime Tests ==========

// Verifies the SelectionHeuristic keeps the resource manager alive even if the
// caller drops its own shared_ptr — this is the core memory-safety guarantee
// the shared_ptr-to-manager design provides.
TEST_F(TestSelectionHeuristic, KeepsResourceManagerAliveAcrossCallerRelease)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xABCD);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));
    EXPECT_CALL(*_mockPlugin, finalize(mockDescriptor)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor)).Times(1);

    const std::weak_ptr<NiceMock<MockHeuristicPluginResourceManager>> weakManager
        = _mockResourceManager;

    SelectionHeuristic heuristic(_mockResourceManager, _policyId);

    // Caller releases its strong reference; the slot should still hold one.
    _mockResourceManager.reset();
    EXPECT_FALSE(weakManager.expired());

    // Operations through the slot still work because the manager is alive.
    EXPECT_TRUE(heuristic.finalize());
}

// ========== Exception Safety Tests ==========

TEST_F(TestSelectionHeuristic, DestructorHandlesExceptionInCleanup)
{
    wireResourceManager();

    auto mockDescriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xFFFF);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor));

    // Destructor should catch and suppress exceptions
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor))
        .WillOnce(::testing::Throw(
            hipdnn_backend::HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Cleanup failed")));

    // Should not throw from destructor
    {
        const SelectionHeuristic heuristic(_mockResourceManager, _policyId);
    }
}

TEST_F(TestSelectionHeuristic, MoveAssignmentHandlesExceptionInCleanup)
{
    wireResourceManager();

    auto mockDescriptor1 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x1001);
    auto mockDescriptor2 = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x2002);

    EXPECT_CALL(*_mockPlugin, createPolicyDescriptor(_mockHandle, _policyId))
        .WillOnce(::testing::Return(mockDescriptor1))
        .WillOnce(::testing::Return(mockDescriptor2));

    // First descriptor cleanup throws during move assignment
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor1))
        .WillOnce(::testing::Throw(std::runtime_error("Cleanup error")));

    // Second descriptor cleanup succeeds
    EXPECT_CALL(*_mockPlugin, destroyPolicyDescriptor(mockDescriptor2)).Times(1);

    {
        SelectionHeuristic heuristic1(_mockResourceManager, _policyId);
        SelectionHeuristic heuristic2(_mockResourceManager, _policyId);

        // Move assignment should not throw even though cleanup of old descriptor throws
        heuristic1 = std::move(heuristic2);
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPlugin.cpp
 * @brief Unit tests for HeuristicPlugin's load-time validation helpers.
 *
 * Workflow / call-sequence / metadata-passthrough behaviors are covered via
 * real plugins in IntegrationHeuristicPlugin.cpp — gmock can only round-trip
 * its own configured returns, which would not exercise any HeuristicPlugin
 * logic.
 */

#include "HipdnnException.hpp"
#include "descriptors/mocks/MockHeuristicPlugin.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using ::testing::NiceMock;
using ::testing::Return;

class TestHeuristicPlugin : public ::testing::Test
{
};

// ========== Plugin Metadata Validation ==========
// HeuristicPlugin::validatePluginMetadata is the load-time gate invoked from
// resolveSymbols(). Each test below pins one specific rejection path that is
// otherwise unreachable without a dedicated test plugin .so.

namespace
{
const std::string_view GOOD_POLICY_NAME = "TestPolicy::Good";
const int64_t GOOD_POLICY_ID = hipdnn_data_sdk::utilities::policyNameToId("TestPolicy::Good");
} // namespace

TEST_F(TestHeuristicPlugin, ValidatePluginMetadataRejectsNonHeuristicPluginType)
{
    const NiceMock<MockHeuristicPlugin> plugin;
    EXPECT_CALL(plugin, type()).WillRepeatedly(Return(HIPDNN_PLUGIN_TYPE_ENGINE));

    EXPECT_THROW(HeuristicPlugin::validatePluginMetadata(plugin), HipdnnException);
}

TEST_F(TestHeuristicPlugin, ValidatePluginMetadataRejectsPolicyIdNameHashMismatch)
{
    const NiceMock<MockHeuristicPlugin> plugin;
    EXPECT_CALL(plugin, type()).WillRepeatedly(Return(HIPDNN_PLUGIN_TYPE_HEURISTIC));
    EXPECT_CALL(plugin, name()).WillRepeatedly(Return("MockPlugin"));
    // Plugin reports policy name "TestPolicy::Good" but tags it with an ID that
    // is NOT policyNameToId("TestPolicy::Good"). validatePluginMetadata must
    // reject this mismatch.
    const int64_t bogusId = GOOD_POLICY_ID ^ int64_t { 0xDEADBEEF };
    EXPECT_CALL(plugin, getAllPolicyIds()).WillRepeatedly(Return(std::vector<int64_t>{bogusId}));
    EXPECT_CALL(plugin, getPolicyName(bogusId)).WillRepeatedly(Return(GOOD_POLICY_NAME));

    EXPECT_THROW(HeuristicPlugin::validatePluginMetadata(plugin), HipdnnException);
}

// ========== Policy ID Buffer Validation ==========
// HeuristicPlugin::validatePolicyIdsBuffer is invoked from getAllPolicyIds()
// after the second (fetch) call into the plugin and gates the lazy enumeration
// cache. Tests below exercise it directly with raw buffers since
// MockHeuristicPlugin overrides getAllPolicyIds() entirely.

TEST_F(TestHeuristicPlugin, ValidatePolicyIdsBufferRejectsZeroPolicyCount)
{
    std::vector<int64_t> ids;
    EXPECT_THROW(HeuristicPlugin::validatePolicyIdsBuffer(0, 0, ids), HipdnnException);
}

TEST_F(TestHeuristicPlugin, ValidatePolicyIdsBufferRejectsCountMismatchBetweenQueries)
{
    std::vector<int64_t> ids = {10, 20, 30};
    EXPECT_THROW(HeuristicPlugin::validatePolicyIdsBuffer(3, 2, ids), HipdnnException);
}

TEST_F(TestHeuristicPlugin, ValidatePolicyIdsBufferRejectsIntraPluginDuplicateIds)
{
    std::vector<int64_t> ids = {42, 42};
    EXPECT_THROW(HeuristicPlugin::validatePolicyIdsBuffer(2, 2, ids), HipdnnException);
}

TEST_F(TestHeuristicPlugin, ValidatePolicyIdsBufferAcceptsValidUniqueIdsAndSorts)
{
    std::vector<int64_t> ids = {30, 10, 20};
    EXPECT_NO_THROW(HeuristicPlugin::validatePolicyIdsBuffer(3, 3, ids));
    EXPECT_EQ(ids, (std::vector<int64_t>{10, 20, 30}));
}

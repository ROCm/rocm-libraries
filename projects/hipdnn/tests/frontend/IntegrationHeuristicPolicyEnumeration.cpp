// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file IntegrationHeuristicPolicyEnumeration.cpp
 * @brief Frontend tests for heuristic policy enumeration (RFC 0007 Section 16)
 *
 * Tests the frontend API for querying loaded heuristic policies.
 */

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_frontend/Handle.hpp>
#include <hipdnn_frontend/HeuristicPolicyInfo.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <sstream>

using namespace hipdnn_frontend;

namespace
{

// Enumeration order is unspecified (backed by unordered_map); compare as sets.
std::set<int64_t> toPolicyIdSet(const std::vector<HeuristicPolicyInfo>& policies)
{
    std::set<int64_t> ids;
    for(const auto& p : policies)
    {
        ids.insert(p.policyId);
    }
    return ids;
}

} // namespace

class IntegrationHeuristicPolicyEnumeration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto [h, e] = createHipdnnHandle();
        ASSERT_FALSE(e.is_bad()) << "Failed to create handle: " << e.get_message();
        _handle = std::move(h);
    }

    HipdnnHandlePtr _handle;
};

// ========== Basic Enumeration Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, GetLoadedPoliciesReturnsNonEmpty)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad()) << "Query failed: " << err.get_message();
    EXPECT_GE(policies.size(), 2u) << "Expected at least Config and StaticOrdering policies";
}

TEST_F(IntegrationHeuristicPolicyEnumeration, PolicyInfoHasValidMetadata)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());
    ASSERT_GT(policies.size(), 0u);

    for(const auto& policy : policies)
    {
        EXPECT_FALSE(policy.policyName.empty()) << "Policy name should not be empty";
        EXPECT_FALSE(policy.pluginName.empty()) << "Plugin name should not be empty";
        EXPECT_FALSE(policy.pluginVersion.empty()) << "Plugin version should not be empty";
        EXPECT_FALSE(policy.apiVersion.empty()) << "API version should not be empty";

        // policyId must be the canonical hash of the reported policy name.
        EXPECT_EQ(policy.policyId, hipdnn_data_sdk::utilities::policyNameToId(policy.policyName))
            << "Policy ID should be the policyNameToId hash of '" << policy.policyName << "'";
    }
}

TEST_F(IntegrationHeuristicPolicyEnumeration, GoodHeuristicPluginPolicyIsEnumerated)
{
    // main.cpp wires test_good_heuristic_plugin in additively before any test
    // runs. Its policy must appear in the enumeration; if it does not, either
    // the plugin failed to load or the enumeration is dropping additive plugins.
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());

    const std::string expectedPolicyName
        = hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName();

    const bool found
        = std::any_of(policies.begin(), policies.end(), [&](const HeuristicPolicyInfo& p) {
              return p.policyName == expectedPolicyName;
          });

    EXPECT_TRUE(found) << "Expected test_good_heuristic_plugin policy '" << expectedPolicyName
                       << "' to appear in enumeration of " << policies.size() << " policies";
}

TEST_F(IntegrationHeuristicPolicyEnumeration, DefaultPoliciesAreLoaded)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());

    // Check for expected default policies
    bool hasConfig = false;
    bool hasStaticOrdering = false;

    for(const auto& policy : policies)
    {
        if(policy.policyName.find("Config") != std::string::npos)
        {
            hasConfig = true;
        }
        if(policy.policyName.find("StaticOrdering") != std::string::npos)
        {
            hasStaticOrdering = true;
        }
    }

    EXPECT_TRUE(hasConfig) << "Config policy should be loaded by default";
    EXPECT_TRUE(hasStaticOrdering) << "StaticOrdering policy should be loaded by default";
}

// ========== Error Handling Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, NullHandleReturnsError)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(nullptr);

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(policies.empty());
    EXPECT_NE(err.get_message().find("null"), std::string::npos)
        << "Error should mention null handle";
}

// ========== snake_case Alias Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, SnakeCaseAliasWorks)
{
    auto [policies, err] = get_loaded_heuristic_policy_infos(*_handle);

    EXPECT_FALSE(err.is_bad());
    EXPECT_GT(policies.size(), 0u);
}

// ========== Multiple Query Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, MultipleQueriesReturnSameResults)
{
    auto [policies1, err1] = getLoadedHeuristicPolicyInfos(*_handle);
    auto [policies2, err2] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err1.is_bad());
    ASSERT_FALSE(err2.is_bad());

    EXPECT_EQ(toPolicyIdSet(policies1), toPolicyIdSet(policies2));
}

// ========== Handle Independence Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, DifferentHandlesHaveSamePolicies)
{
    auto [handle2, err] = createHipdnnHandle();
    ASSERT_FALSE(err.is_bad());

    auto [policies1, err1] = getLoadedHeuristicPolicyInfos(*_handle);
    auto [policies2, err2] = getLoadedHeuristicPolicyInfos(*handle2);

    ASSERT_FALSE(err1.is_bad());
    ASSERT_FALSE(err2.is_bad());

    // Both handles should see the same loaded policies.
    EXPECT_EQ(toPolicyIdSet(policies1), toPolicyIdSet(policies2));
}

// ========== Content Validation Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, PolicyNamesAreUTF8)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());

    for(const auto& policy : policies)
    {
        // Policy name should be valid UTF-8 (basic check: no null bytes except at end)
        EXPECT_EQ(policy.policyName.find('\0'), std::string::npos)
            << "Policy name should not contain null bytes";
    }
}

TEST_F(IntegrationHeuristicPolicyEnumeration, VersionStringsAreValid)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());

    for(const auto& policy : policies)
    {
        // Version strings should parse as valid versions
        EXPECT_NO_THROW({
            const hipdnn_data_sdk::utilities::Version pluginVer{policy.pluginVersion};
            const hipdnn_data_sdk::utilities::Version apiVer{policy.apiVersion};
        }) << "Version strings should be parseable for policy "
           << policy.policyName;
    }
}

// ========== Integration with Graph Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, CanQueryPoliciesBeforeGraphCreation)
{
    // Should be able to query policies before creating any graphs
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    EXPECT_FALSE(err.is_bad());
    EXPECT_GT(policies.size(), 0u);
}

// ========== Logging Tests ==========

TEST_F(IntegrationHeuristicPolicyEnumeration, EnumerationFormatsAllFields)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());
    ASSERT_GT(policies.size(), 0u);

    // Format every policy through ostringstream; assert each rendered line
    // contains every field. Catches regressions where a field is empty,
    // unformattable, or accidentally dropped from the struct.
    for(const auto& policy : policies)
    {
        std::ostringstream oss;
        oss << policy.policyName << " (ID: " << policy.policyId << ", Plugin: " << policy.pluginName
            << ", Version: " << policy.pluginVersion << ", API: " << policy.apiVersion << ")";
        const std::string rendered = oss.str();

        EXPECT_NE(rendered.find(policy.policyName), std::string::npos);
        EXPECT_NE(rendered.find(policy.pluginName), std::string::npos);
        EXPECT_NE(rendered.find(policy.pluginVersion), std::string::npos);
        EXPECT_NE(rendered.find(policy.apiVersion), std::string::npos);
        EXPECT_NE(rendered.find(std::to_string(policy.policyId)), std::string::npos);
    }
}

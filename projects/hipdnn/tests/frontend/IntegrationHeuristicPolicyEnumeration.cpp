// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPolicyEnumeration.cpp
 * @brief Frontend tests for heuristic policy enumeration (RFC 0007 Section 16)
 *
 * Tests the frontend API for querying loaded heuristic policies.
 */

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_frontend/Handle.hpp>
#include <hipdnn_frontend/HeuristicPolicyInfo.hpp>

#include <gtest/gtest.h>

using namespace hipdnn_frontend;

class IntegrationGpuHeuristicPolicyEnumeration : public ::testing::Test
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

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, GetLoadedPoliciesReturnsNonEmpty)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad()) << "Query failed: " << err.get_message();
    EXPECT_GE(policies.size(), 2u) << "Expected at least Config and StaticOrdering policies";
}

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, PolicyInfoHasValidMetadata)
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

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, DefaultPoliciesAreLoaded)
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

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, NullHandleReturnsError)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(nullptr);

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(policies.empty());
    EXPECT_NE(err.get_message().find("null"), std::string::npos)
        << "Error should mention null handle";
}

// ========== snake_case Alias Tests ==========

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, SnakeCaseAliasWorks)
{
    auto [policies, err] = get_loaded_heuristic_policy_infos(*_handle);

    EXPECT_FALSE(err.is_bad());
    EXPECT_GT(policies.size(), 0u);
}

// ========== Multiple Query Tests ==========

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, MultipleQueriesReturnSameResults)
{
    auto [policies1, err1] = getLoadedHeuristicPolicyInfos(*_handle);
    auto [policies2, err2] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err1.is_bad());
    ASSERT_FALSE(err2.is_bad());

    EXPECT_EQ(policies1.size(), policies2.size());

    // Check that policy IDs match
    for(size_t i = 0; i < policies1.size(); ++i)
    {
        EXPECT_EQ(policies1[i].policyId, policies2[i].policyId);
        EXPECT_EQ(policies1[i].policyName, policies2[i].policyName);
    }
}

// ========== Handle Independence Tests ==========

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, DifferentHandlesHaveSamePolicies)
{
    auto [handle2, err] = createHipdnnHandle();
    ASSERT_FALSE(err.is_bad());

    auto [policies1, err1] = getLoadedHeuristicPolicyInfos(*_handle);
    auto [policies2, err2] = getLoadedHeuristicPolicyInfos(*handle2);

    ASSERT_FALSE(err1.is_bad());
    ASSERT_FALSE(err2.is_bad());

    // Both handles should see the same loaded policies
    EXPECT_EQ(policies1.size(), policies2.size());
}

// ========== Content Validation Tests ==========

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, PolicyNamesAreUTF8)
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

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, VersionStringsAreValid)
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

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, CanQueryPoliciesBeforeGraphCreation)
{
    // Should be able to query policies before creating any graphs
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    EXPECT_FALSE(err.is_bad());
    EXPECT_GT(policies.size(), 0u);
}

// ========== Logging Tests ==========

TEST_F(IntegrationGpuHeuristicPolicyEnumeration, EnumerationCanBeLogged)
{
    auto [policies, err] = getLoadedHeuristicPolicyInfos(*_handle);

    ASSERT_FALSE(err.is_bad());

    // Log all policies (visual check in test output)
    std::cout << "Loaded " << policies.size() << " heuristic policies:\n";
    for(const auto& policy : policies)
    {
        std::cout << "  - " << policy.policyName << " (ID: " << policy.policyId
                  << ", Plugin: " << policy.pluginName << ", Version: " << policy.pluginVersion
                  << ", API: " << policy.apiVersion << ")" << '\n';
    }

    SUCCEED();
}

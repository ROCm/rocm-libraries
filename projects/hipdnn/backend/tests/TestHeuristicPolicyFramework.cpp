// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPolicyFramework.cpp
 * @brief Unit tests for the Heuristic Policy Framework
 *
 * Tests cover:
 * - Policy enumeration and metadata via the public hipdnnGetHeuristicPolicy* APIs
 * - Default-policy loading through the resource manager
 *
 * Lower-level coverage (policy order resolution, outer-loop failure handling,
 * empty-policy-list throws) lives in TestEngineHeuristicDescriptorAdditional.cpp,
 * where the mocked plugin manager makes those paths reachable in a unit test.
 */

#include "handle/Handle.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_backend;

class TestHeuristicPolicyFramework : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // hipdnnCreate loads real heuristic plugins (e.g. hipBLASLt in the
        // superbuild) whose initializers probe the device. Skip on no-GPU
        // runners to avoid a hard abort from the plugin's HIP error path.
        SKIP_IF_NO_DEVICES();
        const hipdnnStatus_t status = hipdnnCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_handle, nullptr);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
            _handle = nullptr;
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

// ========== Policy Enumeration Tests ==========

TEST_F(TestHeuristicPolicyFramework, GetHeuristicPolicyCountReturnsNonZero)
{
    size_t numPolicies = 0;
    const hipdnnStatus_t status = hipdnnGetHeuristicPolicyCount_ext(_handle, &numPolicies);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    // At minimum, the StaticOrdering built-in should be loaded.
    EXPECT_GE(numPolicies, 1u);
}

TEST_F(TestHeuristicPolicyFramework, GetHeuristicPolicyInfoReturnsValidData)
{
    size_t numPolicies = 0;
    ASSERT_EQ(hipdnnGetHeuristicPolicyCount_ext(_handle, &numPolicies), HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(numPolicies, 0u);

    // Query first policy (two-call pattern)
    int64_t policyId = -1;
    size_t policyNameLen = 0;
    size_t pluginNameLen = 0;
    size_t pluginVersionLen = 0;
    size_t apiVersionLen = 0;

    // First call: query sizes
    hipdnnStatus_t status = hipdnnGetHeuristicPolicyInfo_ext(_handle,
                                                             0,
                                                             &policyId,
                                                             nullptr,
                                                             &policyNameLen,
                                                             nullptr,
                                                             &pluginNameLen,
                                                             nullptr,
                                                             &pluginVersionLen,
                                                             nullptr,
                                                             &apiVersionLen);

    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(policyId, -1);
    EXPECT_GT(policyNameLen, 0u);
    EXPECT_GT(pluginNameLen, 0u);
    EXPECT_GT(pluginVersionLen, 0u);
    EXPECT_GT(apiVersionLen, 0u);

    // Second call: retrieve strings
    std::vector<char> policyName(policyNameLen);
    std::vector<char> pluginName(pluginNameLen);
    std::vector<char> pluginVersion(pluginVersionLen);
    std::vector<char> apiVersion(apiVersionLen);

    status = hipdnnGetHeuristicPolicyInfo_ext(_handle,
                                              0,
                                              &policyId,
                                              policyName.data(),
                                              &policyNameLen,
                                              pluginName.data(),
                                              &pluginNameLen,
                                              pluginVersion.data(),
                                              &pluginVersionLen,
                                              apiVersion.data(),
                                              &apiVersionLen);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(std::strlen(policyName.data()), 0u);
    EXPECT_GT(std::strlen(pluginName.data()), 0u);
    EXPECT_GT(std::strlen(pluginVersion.data()), 0u);
    EXPECT_GT(std::strlen(apiVersion.data()), 0u);
}

TEST_F(TestHeuristicPolicyFramework, GetHeuristicPolicyInfoOutOfRangeFails)
{
    size_t numPolicies = 0;
    ASSERT_EQ(hipdnnGetHeuristicPolicyCount_ext(_handle, &numPolicies), HIPDNN_STATUS_SUCCESS);

    // Try to query beyond range
    int64_t policyId = -1;
    size_t policyNameLen = 0;
    size_t pluginNameLen = 0;
    size_t pluginVersionLen = 0;
    size_t apiVersionLen = 0;

    const hipdnnStatus_t status = hipdnnGetHeuristicPolicyInfo_ext(_handle,
                                                                   numPolicies + 100,
                                                                   &policyId,
                                                                   nullptr,
                                                                   &policyNameLen,
                                                                   nullptr,
                                                                   &pluginNameLen,
                                                                   nullptr,
                                                                   &pluginVersionLen,
                                                                   nullptr,
                                                                   &apiVersionLen);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

// Policy order resolution (descriptor / env / default), policy decline behavior,
// and "no policy succeeds" failure paths are covered with mocked plugin managers
// in descriptors/TestEngineHeuristicDescriptor.cpp. The StaticOrdering "never
// declines" contract is enforced by the built-in's Finalize implementation and
// exercised in heuristics/TestStaticOrderingBuiltIn.cpp.

// ========== Integration Tests ==========

TEST_F(TestHeuristicPolicyFramework, HeuristicResourceManagerLoadsDefaultPolicies)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // The StaticOrdering built-in is registered at construction time and is the
    // canonical fallback policy.
    EXPECT_GE(policyInfos.size(), 1u);

    bool hasStaticOrdering = false;
    for(const auto& info : policyInfos)
    {
        if(info.policyName.find("StaticOrdering") != std::string::npos)
        {
            hasStaticOrdering = true;
        }
    }

    EXPECT_TRUE(hasStaticOrdering) << "StaticOrdering policy should be loaded";
}

// ========== Negative Tests ==========

TEST_F(TestHeuristicPolicyFramework, GetPolicyCountWithNullHandleFails)
{
    size_t numPolicies = 0;
    const hipdnnStatus_t status = hipdnnGetHeuristicPolicyCount_ext(nullptr, &numPolicies);

    EXPECT_NE(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(TestHeuristicPolicyFramework, GetPolicyCountWithNullPointerFails)
{
    const hipdnnStatus_t status = hipdnnGetHeuristicPolicyCount_ext(_handle, nullptr);

    EXPECT_NE(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(TestHeuristicPolicyFramework, GetPolicyInfoWithNullLengthPointersFails)
{
    int64_t policyId = -1;

    // All length pointers are required (not nullptr)
    const hipdnnStatus_t status = hipdnnGetHeuristicPolicyInfo_ext(_handle,
                                                                   0,
                                                                   &policyId,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr);

    EXPECT_NE(status, HIPDNN_STATUS_SUCCESS);
}

// Note: gtest provides main(), do not define it here

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdBuiltIn.cpp
 * @brief Tests for the SelectionHeuristic::UHD built-in policy.
 *
 * The built-in exposes a function-pointer table wrapped by HeuristicPlugin via
 * createBuiltIn. These tests exercise the C ABI entry points through the wrapper.
 */

#include "heuristics/uhd/UhdBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>

#include <vector>

using hipdnn_backend::heuristics::uhd::populateFunctionTable;
using hipdnn_backend::plugin::HeuristicPlugin;
using hipdnn_backend::plugin::HeuristicPluginFunctionTable;

namespace
{

const int64_t UHD_POLICY_ID = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::UHD");

class TestUhdBuiltIn : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _plugin = HeuristicPlugin::createBuiltIn(populateFunctionTable(), "built-in:UHD-test");
        _handle = _plugin->createHandle();
        ASSERT_NE(_handle, nullptr);
        _desc = _plugin->createPolicyDescriptor(_handle, UHD_POLICY_ID);
        ASSERT_NE(_desc, nullptr);
    }

    void TearDown() override
    {
        if(_desc != nullptr)
        {
            _plugin->destroyPolicyDescriptor(_desc);
        }
        if(_handle != nullptr)
        {
            _plugin->destroyHandle(_handle);
        }
    }

    std::shared_ptr<HeuristicPlugin> _plugin;
    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

// Convenience: grab the raw function table for direct C-ABI tests.
const HeuristicPluginFunctionTable& uhdAbi()
{
    static const HeuristicPluginFunctionTable s_funcs = populateFunctionTable();
    return s_funcs;
}

// ========== Plugin metadata ==========

TEST_F(TestUhdBuiltIn, ReportsHeuristicPluginType)
{
    EXPECT_EQ(_plugin->type(), HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST_F(TestUhdBuiltIn, EnumeratesSingleUhdPolicy)
{
    auto policyIds = _plugin->getAllPolicyIds();
    ASSERT_EQ(policyIds.size(), 1u);
    EXPECT_EQ(policyIds[0], UHD_POLICY_ID);
}

TEST_F(TestUhdBuiltIn, GetPolicyNameReturnsUhd)
{
    auto name = _plugin->getPolicyName(UHD_POLICY_ID);
    EXPECT_EQ(name, "SelectionHeuristic::UHD");
}

TEST_F(TestUhdBuiltIn, GetPolicyNameThrowsOnBadId)
{
    EXPECT_THROW(_plugin->getPolicyName(9999), std::runtime_error);
}

// ========== Handle lifecycle ==========

TEST_F(TestUhdBuiltIn, CreateAndDestroyHandle)
{
    auto handle = _plugin->createHandle();
    EXPECT_NE(handle, nullptr);
    _plugin->destroyHandle(handle);
}

// ========== Policy descriptor lifecycle ==========

TEST_F(TestUhdBuiltIn, CreatePolicyDescriptorSucceeds)
{
    auto desc = _plugin->createPolicyDescriptor(_handle, UHD_POLICY_ID);
    EXPECT_NE(desc, nullptr);
    _plugin->destroyPolicyDescriptor(desc);
}

TEST_F(TestUhdBuiltIn, CreatePolicyDescriptorBadIdThrows)
{
    EXPECT_THROW(_plugin->createPolicyDescriptor(_handle, 9999), std::runtime_error);
}

// ========== Engine IDs ==========

TEST_F(TestUhdBuiltIn, SetEngineIdsSucceeds)
{
    std::vector<int64_t> engineIds = {100, 200, 300};
    // Should not throw
    _plugin->setEngineIds(_desc, engineIds);
}

TEST_F(TestUhdBuiltIn, SetEmptyEngineIds)
{
    std::vector<int64_t> emptyIds;
    _plugin->setEngineIds(_desc, emptyIds);
}

// ========== Finalize (stub behavior) ==========

TEST_F(TestUhdBuiltIn, FinalizeDeclinesDueToStub)
{
    // Currently UHD declines because RFC-0017 integration is pending
    std::vector<int64_t> engineIds = {100, 200, 300};
    _plugin->setEngineIds(_desc, engineIds);

    bool applied = _plugin->finalize(_desc);
    // UHD stub returns outApplied = 0
    EXPECT_FALSE(applied);
}

TEST_F(TestUhdBuiltIn, FinalizeWithNoEnginesDeclines)
{
    std::vector<int64_t> emptyIds;
    _plugin->setEngineIds(_desc, emptyIds);

    bool applied = _plugin->finalize(_desc);
    EXPECT_FALSE(applied);
}

// ========== Get sorted engine IDs ==========

TEST_F(TestUhdBuiltIn, GetSortedEngineIdsAfterFinalize)
{
    std::vector<int64_t> engineIds = {100, 200, 300};
    _plugin->setEngineIds(_desc, engineIds);
    _plugin->finalize(_desc);

    // Since UHD declines, sorted list should be empty
    auto sorted = _plugin->getSortedEngineIds(_desc);
    EXPECT_TRUE(sorted.empty());
}

// ========== C-ABI rejection tests ==========

TEST(TestUhdBuiltInCabi, HandleCreateNullOutput)
{
    auto status = uhdAbi().handleCreate(nullptr);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestUhdBuiltInCabi, HandleDestroyNullHandle)
{
    auto status = uhdAbi().handleDestroy(nullptr);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestUhdBuiltInCabi, PolicyDescriptorCreateNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    auto status = uhdAbi().policyDescriptorCreate(nullptr, UHD_POLICY_ID, &desc);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestUhdBuiltInCabi, PolicySetEngineIdsNullDescriptor)
{
    std::vector<int64_t> ids = {1, 2, 3};
    auto status = uhdAbi().policySetEngineIds(nullptr, ids.data(), ids.size());
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestUhdBuiltInCabi, PolicyFinalizeNullDescriptor)
{
    int32_t applied = 0;
    auto status = uhdAbi().policyFinalize(nullptr, &applied);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestUhdBuiltInCabi, PolicyFinalizeNullOutput)
{
    // Need a real descriptor to test null output
    hipdnnHeuristicHandle_t handle = nullptr;
    auto status = uhdAbi().handleCreate(&handle);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    status = uhdAbi().policyDescriptorCreate(handle, UHD_POLICY_ID, &desc);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    status = uhdAbi().policyFinalize(desc, nullptr);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    uhdAbi().policyDescriptorDestroy(desc);
    uhdAbi().handleDestroy(handle);
}

TEST(TestUhdBuiltInCabi, GetSortedEngineIdsNotFinalized)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    auto status = uhdAbi().handleCreate(&handle);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    status = uhdAbi().policyDescriptorCreate(handle, UHD_POLICY_ID, &desc);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    size_t numEngines = 10;
    status = uhdAbi().policyGetSortedEngineIds(desc, nullptr, &numEngines);
    // Should fail because not finalized
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);

    uhdAbi().policyDescriptorDestroy(desc);
    uhdAbi().handleDestroy(handle);
}

} // namespace

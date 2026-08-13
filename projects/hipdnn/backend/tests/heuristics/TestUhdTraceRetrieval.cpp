// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdTraceRetrieval.cpp
 * @brief Tests for UHD trace retrieval API (RFC 0019 §13).
 *
 * Verifies that SelectionTrace can be retrieved programmatically after
 * finalize via hipdnnHeuristicPolicyGetTrace.
 */

#include "heuristics/uhd/EngineRegistry.hpp"
#include "heuristics/uhd/SelectionEngine.hpp"
#include "heuristics/uhd/UhdBuiltIn.hpp"

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_test_sdk/ScopedTemporaryFile.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>

using namespace hipdnn_backend::heuristics::uhd;

namespace
{

class TestUhdTraceRetrieval : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear registry before each test
        EngineRegistry::instance().clear();

        // Populate function table
        funcs = populateFunctionTable();
    }

    void TearDown() override
    {
        EngineRegistry::instance().clear();
    }

    hipdnn_backend::plugin::HeuristicPluginFunctionTable funcs;
};

TEST_F(TestUhdTraceRetrieval, GetTraceAfterFinalize)
{
    // Register a simple static_order engine
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "test_uhd_trace";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};

    // Add one candidate
    KernelCandidate candidate;
    candidate.kernelId = 1;
    candidate.metadata["priority"] = 10.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(funcs.handleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Create policy descriptor
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    const int64_t policyId = 0x1122334455667788LL; // UHD policy ID
    ASSERT_EQ(funcs.policyDescriptorCreate(handle, policyId, &desc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs
    int64_t engineIds[] = {100};
    ASSERT_EQ(funcs.policySetEngineIds(desc, engineIds, 1), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Finalize
    int32_t applied = 0;
    ASSERT_EQ(funcs.policyFinalize(desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(applied, 1);

    // Retrieve trace
    const char* traceJson = nullptr;
    ASSERT_EQ(funcs.policyGetTrace(desc, 100, &traceJson), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson, nullptr);

    // Parse JSON
    nlohmann::json trace = nlohmann::json::parse(traceJson);

    // Verify trace fields
    EXPECT_EQ(trace["uhd_id"], "test_uhd_trace");
    EXPECT_EQ(trace["adapter_type"], "static_order");
    EXPECT_FALSE(trace["used_model"]); // static_order doesn't use a model

    // Cleanup
    ASSERT_EQ(funcs.policyDescriptorDestroy(desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(funcs.handleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestUhdTraceRetrieval, GetTraceNotFinalizedReturnsError)
{
    // Create handle
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(funcs.handleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Create policy descriptor
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    const int64_t policyId = 0x1122334455667788LL;
    ASSERT_EQ(funcs.policyDescriptorCreate(handle, policyId, &desc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Try to get trace before finalize
    const char* traceJson = nullptr;
    EXPECT_EQ(funcs.policyGetTrace(desc, 100, &traceJson),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);

    // Cleanup
    ASSERT_EQ(funcs.policyDescriptorDestroy(desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(funcs.handleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestUhdTraceRetrieval, GetTraceForUnknownEngineReturnsNotSupported)
{
    // Register engine 100
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "test_uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(funcs.handleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Create policy descriptor
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    const int64_t policyId = 0x1122334455667788LL;
    ASSERT_EQ(funcs.policyDescriptorCreate(handle, policyId, &desc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Set engine IDs and finalize
    int64_t engineIds[] = {100};
    ASSERT_EQ(funcs.policySetEngineIds(desc, engineIds, 1), HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = 0;
    ASSERT_EQ(funcs.policyFinalize(desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Try to get trace for engine that wasn't finalized (engine 200)
    const char* traceJson = nullptr;
    EXPECT_EQ(funcs.policyGetTrace(desc, 200, &traceJson), HIPDNN_PLUGIN_STATUS_NOT_SUPPORTED);

    // Cleanup
    ASSERT_EQ(funcs.policyDescriptorDestroy(desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(funcs.handleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestUhdTraceRetrieval, GetTraceWithModelAdapter)
{
    // This test demonstrates trace retrieval for a model-based adapter
    // when model artifacts become available.
    //
    // For now, we test with static_order since tree_data/onnx/custom_library
    // require actual model files.

    EngineEntry entry;
    entry.engineId = 200;
    entry.uhdConfig.uhdId = "model_uhd_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};

    KernelCandidate candidate;
    candidate.kernelId = 5;
    candidate.metadata["priority"] = 20.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle and descriptor
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(funcs.handleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    const int64_t policyId = 0x1122334455667788LL;
    ASSERT_EQ(funcs.policyDescriptorCreate(handle, policyId, &desc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int64_t engineIds[] = {200};
    ASSERT_EQ(funcs.policySetEngineIds(desc, engineIds, 1), HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = 0;
    ASSERT_EQ(funcs.policyFinalize(desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Get trace
    const char* traceJson = nullptr;
    ASSERT_EQ(funcs.policyGetTrace(desc, 200, &traceJson), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson, nullptr);

    // Parse and verify
    nlohmann::json trace = nlohmann::json::parse(traceJson);
    EXPECT_EQ(trace["uhd_id"], "model_uhd_test");
    EXPECT_TRUE(trace.contains("adapter_type"));

    // Cleanup
    ASSERT_EQ(funcs.policyDescriptorDestroy(desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(funcs.handleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(TestUhdTraceRetrieval, TraceJsonCachedAcrossMultipleCalls)
{
    // Verify that the JSON string is cached and remains valid

    EngineEntry entry;
    entry.engineId = 300;
    entry.uhdConfig.uhdId = "cached_trace_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"id"};
    EngineRegistry::instance().registerEngine(std::move(entry));

    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_EQ(funcs.handleCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    const int64_t policyId = 0x1122334455667788LL;
    ASSERT_EQ(funcs.policyDescriptorCreate(handle, policyId, &desc),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    int64_t engineIds[] = {300};
    ASSERT_EQ(funcs.policySetEngineIds(desc, engineIds, 1), HIPDNN_PLUGIN_STATUS_SUCCESS);

    int32_t applied = 0;
    ASSERT_EQ(funcs.policyFinalize(desc, &applied), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Get trace first time
    const char* traceJson1 = nullptr;
    ASSERT_EQ(funcs.policyGetTrace(desc, 300, &traceJson1), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson1, nullptr);

    // Get trace second time (should be cached)
    const char* traceJson2 = nullptr;
    ASSERT_EQ(funcs.policyGetTrace(desc, 300, &traceJson2), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson2, nullptr);

    // Pointers should be identical (same cached string)
    EXPECT_EQ(traceJson1, traceJson2);

    // Content should be identical
    EXPECT_STREQ(traceJson1, traceJson2);

    // Cleanup
    ASSERT_EQ(funcs.policyDescriptorDestroy(desc), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(funcs.handleDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

#include "core/EngineApiFloor.hpp"
#include "mocks/StampedGraph.hpp"

using hip_kernel_provider::StampedGraph;
using hip_kernel_provider::core::needsNewerThanCompiledInEngineApi;

TEST(TestEngineApiFloor, CompiledInEnginesUnderstandGraphsUpToRuntimePassByValue)
{
    EXPECT_FALSE(needsNewerThanCompiledInEngineApi(StampedGraph(std::nullopt)));
    EXPECT_FALSE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE)));
    EXPECT_FALSE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION)));
    EXPECT_FALSE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION)));
}

TEST(TestEngineApiFloor, RaggedAlignedAndTokenUnitGraphsNeedANewerApi)
{
    EXPECT_TRUE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION)));
    EXPECT_TRUE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION)));
    EXPECT_TRUE(needsNewerThanCompiledInEngineApi(
        StampedGraph(hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION)));
}

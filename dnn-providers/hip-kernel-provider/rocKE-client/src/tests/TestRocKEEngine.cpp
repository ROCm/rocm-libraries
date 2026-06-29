// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RocKEHandle.hpp"
#include "engines/RocKEEngine.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

TEST(TestRocKEEngine, IdReturnsRegisteredRocKEEngineId)
{
    const rocke_client::RocKEEngine engine;

    EXPECT_EQ(engine.id(), hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}

TEST(TestRocKEEngine, IsApplicableRejectsInvalidGraph)
{
    rocke_client::RocKEEngine engine;
    rocke_client::RocKEHandle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);

    EXPECT_FALSE(engine.isApplicable(handle, invalidGraph));
}

TEST(TestRocKEEngine, WorkspaceQueryRejectsSkeletonEngine)
{
    rocke_client::RocKEEngine engine;
    rocke_client::RocKEHandle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);

    EXPECT_THROW(static_cast<void>(engine.getMaxWorkspaceSize(handle, invalidGraph, invalidConfig)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRocKEEngine, ExecutionContextCreationRejectsSkeletonEngine)
{
    rocke_client::RocKEEngine engine;
    rocke_client::RocKEHandle handle;
    rocke_client::RocKEContext context;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);

    EXPECT_THROW(engine.initializeExecutionContext(handle, invalidGraph, invalidConfig, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

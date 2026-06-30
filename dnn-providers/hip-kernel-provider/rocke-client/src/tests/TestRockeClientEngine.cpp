// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RockeClientHandle.hpp"
#include "engines/RockeClientEngine.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

TEST(TestRockeClientEngine, IdReturnsRegisteredRockeEngineId)
{
    const rocke_client::RockeClientEngine engine;

    EXPECT_EQ(engine.id(), hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}

TEST(TestRockeClientEngine, IsApplicableRejectsInvalidGraph)
{
    rocke_client::RockeClientEngine engine;
    rocke_client::RockeClientHandle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);

    EXPECT_FALSE(engine.isApplicable(handle, invalidGraph));
}

TEST(TestRockeClientEngine, GetDetailsBuildsAndReleasesBuffer)
{
    rocke_client::RockeClientEngine engine;
    rocke_client::RockeClientHandle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraph(nullptr, 0);

    hipdnnPluginConstData_t details{nullptr, 0};
    engine.getDetails(handle, opGraph, details);

    ASSERT_NE(details.ptr, nullptr);
    EXPECT_GT(details.size, 0u);

    // The handle owns the backing DetachedBuffer until explicitly removed; releasing it must not
    // leak (validated under sanitizer-enabled CI).
    handle.removeEngineDetailsDetachedBuffer(details.ptr);
}

TEST(TestRockeClientEngine, WorkspaceQueryRejectsSkeletonEngine)
{
    rocke_client::RockeClientEngine engine;
    rocke_client::RockeClientHandle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);

    EXPECT_THROW(static_cast<void>(engine.getMaxWorkspaceSize(handle, invalidGraph, invalidConfig)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRockeClientEngine, ExecutionContextCreationRejectsSkeletonEngine)
{
    rocke_client::RockeClientEngine engine;
    rocke_client::RockeClientHandle handle;
    rocke_client::RockeClientContext context;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper invalidGraph(nullptr, 0);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);

    EXPECT_THROW(engine.initializeExecutionContext(handle, invalidGraph, invalidConfig, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

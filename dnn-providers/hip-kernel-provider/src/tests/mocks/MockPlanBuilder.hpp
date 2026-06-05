// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

namespace hip_kernel_provider
{

class MockPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const Handle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const Handle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const Settings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionSettings,
                (const Handle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 Settings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const Handle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 Context& executionContext),
                (const, override));

    MOCK_METHOD((std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>),
                getCustomKnobs,
                (const Handle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
};

} // namespace hip_kernel_provider

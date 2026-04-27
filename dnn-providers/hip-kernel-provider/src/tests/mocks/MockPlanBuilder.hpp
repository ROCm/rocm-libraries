// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"

namespace hip_kernel_provider
{

class MockPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipKernelHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipKernelHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const HipKernelSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionSettings,
                (const HipKernelHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipKernelSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipKernelHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipKernelContext& executionContext),
                (const, override));

    MOCK_METHOD((std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>),
                getCustomKnobs,
                (const HipKernelHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
};

} // namespace hip_kernel_provider

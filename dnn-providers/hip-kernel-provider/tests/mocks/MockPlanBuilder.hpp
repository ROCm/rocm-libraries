// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"

namespace hip_kernel_plugin
{

class MockPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<HipdnnHipKernelHandle,
                                                               HipdnnHipKernelSettings,
                                                               HipdnnHipKernelContext>
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const HipdnnHipKernelSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionSettings,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnHipKernelSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnHipKernelContext& executionContext),
                (const, override));

    MOCK_METHOD((std::vector<hipdnn_data_sdk::data_objects::KnobT>),
                getCustomKnobs,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
};

}

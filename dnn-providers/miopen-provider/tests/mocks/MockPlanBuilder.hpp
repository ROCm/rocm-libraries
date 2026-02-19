// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include "engines/plans/PlanBuilderInterface.hpp"

namespace miopen_plugin
{

class MockPlanBuilder : public IPlanBuilder
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const MiopenExecutionSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionSettings,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 MiopenExecutionSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 HipdnnEnginePluginExecutionContext& executionContext),
                (const, override));

    MOCK_METHOD((std::vector<hipdnn_data_sdk::data_objects::KnobT>),
                getCustomKnobs,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
};

}
